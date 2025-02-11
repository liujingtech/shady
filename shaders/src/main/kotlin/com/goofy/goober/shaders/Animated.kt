package com.goofy.goober.shaders

import android.graphics.RuntimeShader

/**
 * Shadertoy's default shader
 */
val AurorasShader = RuntimeShader(
"""
uniform float2 iResolution;
uniform float iMouseX;
uniform float iMouseY;
uniform float iMouseZ;
uniform float iTime;
uniform shader iImage1; 
uniform float iMouse;
// 替换 S 宏为函数
float S(float x, float y, float z) {
    return smoothstep(x, y, z);
}

// 替换 B 宏为函数
float B(float a, float b, float edge, float t) {
    return S(a - edge, a + edge, t) * S(b + edge, b - edge, t);
}

// 替换 sat 宏为函数
float sat(float x) {
    return clamp(x, 0., 1.);
}

// 替换颜色宏为变量
const vec3 streetLightCol = vec3(1., .7, .3);
const vec3 headLightCol = vec3(.8, .8, 1.);
const vec3 tailLightCol = vec3(1., .1, .1);

// 替换 HIGH_QUALITY 宏为变量
const bool HIGH_QUALITY = true;

// 替换 CAM_SHAKE 宏为变量
const float CAM_SHAKE = 1.;

// 替换 LANE_BIAS 宏为变量
const float LANE_BIAS = .5;

// 替换 RAIN 宏，假设它代表雨强度，用变量表示
uniform float RAIN;



vec3 N13(float p) {
    //  from DAVE HOSKINS
   vec3 p3 = fract(vec3(p) * vec3(.1031,.11369,.13787));
   p3 += dot(p3, p3.yzx + 19.19);
   return fract(vec3((p3.x + p3.y)*p3.z, (p3.x+p3.z)*p3.y, (p3.y+p3.z)*p3.x));
}

vec4 N14(float t) {
	return fract(sin(t*vec4(123., 1024., 1456., 264.))*vec4(6547., 345., 8799., 1564.));
}
float N(float t) {
    return fract(sin(t*12345.564)*7658.76);
}

float Saw(float b, float t) {
	return S(0., b, t)*S(1., b, t);
}


vec2 DropLayer2(vec2 uv, float t) {
    vec2 UV = uv;
    
    uv.y += t*0.75;
    vec2 a = vec2(6., 1.);
    vec2 grid = a*2.;
    vec2 id = floor(uv*grid);
    
    float colShift = N(id.x); 
    uv.y += colShift;
    
    id = floor(uv*grid);
    vec3 n = N13(id.x*35.2+id.y*2376.1);
    vec2 st = fract(uv*grid)-vec2(.5, 0);
    
    float x = n.x-.5;
    
    float y = UV.y*20.;
    float wiggle = sin(y+sin(y));
    x += wiggle*(.5-abs(x))*(n.z-.5);
    x *= .7;
    float ti = fract(t+n.z);
    y = (Saw(.85, ti)-.5)*.9+.5;
    vec2 p = vec2(x, y);
    
    float d = length((st-p)*a.yx);
    
    float mainDrop = S(.4, .0, d);
    
    float r = sqrt(S(1., y, st.y));
    float cd = abs(st.x-x);
    float trail = S(.23*r, .15*r*r, cd);
    float trailFront = S(-.02, .02, st.y-y);
    trail *= trailFront*r*r;
    
    y = UV.y;
    float trail2 = S(.2*r, .0, cd);
    float droplets = max(0., (sin(y*(1.-y)*120.)-st.y))*trail2*trailFront*n.z;
    y = fract(y*10.)+(st.y-.5);
    float dd = length(st-vec2(x, y));
    droplets = S(.3, 0., dd);
    float m = mainDrop+droplets*r*trailFront;
    
    //m += st.x>a.y*.45 || st.y>a.x*.165 ? 1.2 : 0.;
    return vec2(m, trail);
}

float StaticDrops(vec2 uv, float t) {
	uv *= 40.;
    
    vec2 id = floor(uv);
    uv = fract(uv)-.5;
    vec3 n = N13(id.x*107.45+id.y*3543.654);
    vec2 p = (n.xy-.5)*.7;
    float d = length(uv-p);
    
    float fade = Saw(.025, fract(t+n.z));
    float c = S(.3, 0., d)*fract(n.z*10.)*fade;
    return c;
}

vec2 Drops(vec2 uv, float t, float l0, float l1, float l2) {
    float s = StaticDrops(uv, t)*l0; 
    vec2 m1 = DropLayer2(uv, t)*l1;
    vec2 m2 = DropLayer2(uv*1.85, t)*l2;
    
    float c = s+m1.x+m2.x;
    c = S(.3, 1., c);
    
    return vec2(c, max(m1.y*l0, m2.y*l1));
}

vec4 main( in vec2 fragCoord )
{
	vec2 uv = (fragCoord.xy-.5*iResolution.xy) / iResolution.y;
    vec2 UV = fragCoord.xy/iResolution.xy;
//    vec3 M = vec3(iMouseX,iMouseY,iMouseZ)/iResolution.xyz;
    vec3 M =vec3(iMouseX,iMouseY,iMouseZ)/vec3(iResolution.xy,1);
    float T = iTime+M.x*2.;
    
    T = mod(iTime, 102.);
    T = mix(T, M.x*102., M.z>0.?1.:0.);
    
    
    float t = T*.2;
    
    float rainAmount = iMouseZ>0. ? M.y : sin(T*.05)*.3+.7;
//    float rainAmount = M.y;
    
    float maxBlur = mix(3., 6., rainAmount);
    float minBlur = 2.;
    
    float story = 0.;
    float heart = 0.;
    
    story = S(0., 70., T);
    
    t = min(1., T/70.);						// remap drop time so it goes slower when it freezes
    t = 1.-t;
    t = (1.-t*t)*70.;
    
    float zoom= mix(.3, 1.2, story);		// slowly zoom out
    uv *=zoom;
    minBlur = 4.+S(.5, 1., story)*3.;		// more opaque glass towards the end
    maxBlur = 6.+S(.5, 1., story)*1.5;
    
    vec2 hv = uv-vec2(.0, -.1);				// build heart
    hv.x *= .5;
    float s = S(110., 70., T);				// heart gets smaller and fades towards the end
    hv.y-=sqrt(abs(hv.x))*.5*s;
    heart = length(hv);
    heart = S(.4*s, .2*s, heart)*s;
    rainAmount = heart;						// the rain is where the heart is
    
    maxBlur-=heart;							// inside the heart slighly less foggy
    uv *= 1.5;								// zoom out a bit more
    t *= .25;
    UV = (UV-.5)*(.9+zoom*.1)+.5;
    
    float staticDrops = S(-.5, 1., rainAmount)*2.;
    float layer1 = S(.25, .75, rainAmount);
    float layer2 = S(.0, .5, rainAmount);
    
    
    vec2 c = Drops(uv, t, staticDrops, layer1, layer2);
    	vec2 e = vec2(.001, 0.);
    	float cx = Drops(uv+e, t, staticDrops, layer1, layer2).x;
    	float cy = Drops(uv+e.yx, t, staticDrops, layer1, layer2).x;
    	vec2 n = vec2(cx-c.x, cy-c.x);		// expensive normals
    
    
    n *= 1.-S(60., 85., T);
    c.y *= 1.-S(80., 100., T)*.8;
    
    float focus = mix(maxBlur-c.y, minBlur, S(.1, .2, c.x));
//  float2 scale = iImageResolution.xy / iResolution.xy;
  float2 scale =  iResolution.xy / iResolution.xy;
    vec3 col = iImage1.eval(fragCoord * scale).rgb;
    
    
    //#ifdef USE_POST_PROCESSING
    t = (T+3.)*.5;										// make time sync with first lightnoing
    float colFade = sin(t*.2)*.5+.5+story;
    col *= mix(vec3(1.), vec3(.8, .9, 1.3), colFade);	// subtle color shift
    float fade = S(0., 10., T);							// fade in at the start
    float lightning = sin(t*sin(t*10.));				// lighting flicker
    lightning *= pow(max(0., sin(t+sin(t))), 10.);		// lightning flash
    col *= 1.+lightning*fade*mix(1., .1, story*story);	// composite lightning
    col *= 1.-dot(UV-=.5, UV);							// vignette
    											
    //#ifdef HAS_HEART
    	col = mix(pow(col, vec3(1.2)), col, heart);
    	fade *= S(102., 97., T);
    //#endif
    
    //col *= fade;										// composite start and end fade
    //#endif
    
    //col = vec3(heart);
    return vec4(col, 1.);
}

""".trimIndent()
)
/**
 * Shadertoy's default shader
 */
val StarryShader = RuntimeShader(
"""// CC0: Starry planes
//  Revisited ye olde "plane-marcher".
//  A simple result that I think turned out pretty nice
uniform float2 iResolution;
uniform float2 iMouse;
uniform float iTime;

mat2 ROT(float a) {
    // 创建并返回旋转矩阵
    return mat2(cos(a), sin(a), -sin(a), cos(a));
}

const float
  pi        = acos(-1.)
, tau       = 2.*pi
, planeDist = .5
, furthest  = 16.
, fadeFrom  = 8.
;

const vec2 
  pathA = vec2(.31, .41)
, pathB = vec2(1.0,sqrt(0.5))
;

const vec4 
  U = vec4(0, 1, 2, 3)
  ;
  
// License: Unknown, author: Matt Taylor (https://github.com/64), found: https://64.github.io/tonemapping/
vec3 aces_approx(vec3 v) {
  v = max(v, 0.0);
  v *= 0.6;
  float a = 2.51;
  float b = 0.03;
  float c = 2.43;
  float d = 0.59;
  float e = 0.14;
  return clamp((v*(a*v+b))/(v*(c*v+d)+e), 0.0, 1.0);
}

vec3 offset(float z) {
  return vec3(pathB*sin(pathA*z), z);
}

vec3 doffset(float z) {
  return vec3(pathA*pathB*cos(pathA*z), 1.0);
}

vec3 ddoffset(float z) {
  return vec3(-pathA*pathA*pathB*sin(pathA*z), 0.0);
}

vec4 alphaBlend(vec4 back, vec4 front) {
  // Based on: https://en.wikipedia.org/wiki/Alpha_compositing
  float w = front.w + back.w*(1.0-front.w);
  vec3 xyz = (front.xyz*front.w + back.xyz*back.w*(1.0-front.w))/w;
  return w > 0.0 ? vec4(xyz, w) : vec4(0.0);
}

// License: MIT, author: Inigo Quilez, found: https://www.iquilezles.org/www/articles/smin/smin.htm
float pmin(float a, float b, float k) {
  float h = clamp(0.5+0.5*(b-a)/k, 0.0, 1.0);
  return mix(b, a, h) - k*h*(1.0-h);
}

float pmax(float a, float b, float k) {
  return -pmin(-a, -b, k);
}

float pabs(float a, float k) {
  return -pmin(a, -a, k);
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/articles/distfunctions2d/
//   Slightly tweaked to round the inner corners
float star5(vec2 p, float r, float rf, float sm) {
  p = -p;
  const vec2 k1 = vec2(0.809016994375, -0.587785252292);
  const vec2 k2 = vec2(-k1.x,k1.y);
  p.x = abs(p.x);
  p -= 2.0*max(dot(k1,p),0.0)*k1;
  p -= 2.0*max(dot(k2,p),0.0)*k2;
  p.x = pabs(p.x, sm);
  p.y -= r;
  vec2 ba = rf*vec2(-k1.y,k1.x) - vec2(0,1);
  float h = clamp( dot(p,ba)/dot(ba,ba), 0.0, r );
  return length(p-ba*h) * sign(p.y*ba.x-p.x*ba.y);
}

vec3 palette(float n) {
  return 0.5+0.5*sin(vec3(0.,1.,2.)+n);
}

vec4 plane(vec3 ro, vec3 rd, vec3 pp, vec3 npp, float pd, vec3 cp, vec3 off, float n) {

  float aa = 3.*pd*distance(pp.xy, npp.xy);
  vec4 col = vec4(0.);
  vec2 p2 = pp.xy;
  p2 -= offset(pp.z).xy;
  vec2 doff   = ddoffset(pp.z).xz;
  vec2 ddoff  = doffset(pp.z).xz;
  float dd = dot(doff, ddoff);
  p2 *= ROT(dd*pi*5.);

  float d0 = star5(p2, 0.45, 1.6,0.2)-0.02;
  float d1 = d0-0.01;
  float d2 = length(p2);
  const float colp = pi*100.;
  float colaa = aa*200.;
  
  col.xyz = palette(0.5*n+2.*d2)*mix(0.5/(d2*d2), 1., smoothstep(-0.5+colaa, 0.5+colaa, sin(d2*colp)))/max(3.*d2*d2, 1E-1);
  col.xyz = mix(col.xyz, vec3(2.), smoothstep(aa, -aa, d1)); 
  col.w = smoothstep(aa, -aa, -d0);
  return col;

}

vec3 color(vec3 ww, vec3 uu, vec3 vv, vec3 ro, vec2 p) {
  float lp = length(p);
  vec2 np = p + 1./iResolution.xy;
  float rdd = 2.0-0.25;
  
  vec3 rd = normalize(p.x*uu + p.y*vv + rdd*ww);
  vec3 nrd = normalize(np.x*uu + np.y*vv + rdd*ww);

  float nz = floor(ro.z / planeDist);

  vec4 acol = vec4(0.0);

  vec3 aro = ro;
  float apd = 0.0;

  for (float i = 1.; i <= furthest; ++i) {
    if ( acol.w > 0.95) {
      // Debug col to see when exiting
      // acol.xyz = palette(i); 
      break;
    }
    float pz = planeDist*nz + planeDist*i;

    float lpd = (pz - aro.z)/rd.z;
    float npd = (pz - aro.z)/nrd.z;
    float cpd = (pz - aro.z)/ww.z;

    {
      vec3 pp = aro + rd*lpd;
      vec3 npp= aro + nrd*npd;
      vec3 cp = aro+ww*cpd;

      apd += lpd;

      vec3 off = offset(pp.z);

      float dz = pp.z-ro.z;
      float fadeIn = smoothstep(planeDist*furthest, planeDist*fadeFrom, dz);
      float fadeOut = smoothstep(0., planeDist*.1, dz);
      float fadeOutRI = smoothstep(0., planeDist*1.0, dz);

      float ri = mix(1.0, 0.9, fadeOutRI*fadeIn);

      vec4 pcol = plane(ro, rd, pp, npp, apd, cp, off, nz+i);

      pcol.w *= fadeOut*fadeIn;
      acol = alphaBlend(pcol, acol);
      aro = pp;
    }
    
  }

  return acol.xyz*acol.w;

}

vec4 main( in float2 fragCoord )
        {
  vec2 r = iResolution.xy, q = fragCoord/r, pp = -1.0+2.0*q, p = pp;
  p.x *= r.x/r.y;

  float tm  = planeDist*iTime;

  vec3 ro   = offset(tm);
  vec3 dro  = doffset(tm);
  vec3 ddro = ddoffset(tm);

  vec3 ww = normalize(dro);
  vec3 uu = normalize(cross(U.xyx+ddro, ww));
  vec3 vv = cross(ww, uu);
  
  vec3 col = color(ww, uu, vv, ro, p);
  col = aces_approx(col);
  col = sqrt(col);
  return vec4(col, 1);
}


"""
)

/**
 * Shadertoy's default shader
 */
val SkyShader = RuntimeShader(
    """
        const float cloudscale = 1.1;
        const float speed = 0.03;
        const float clouddark = 0.5;
        const float cloudlight = 0.3;
        const float cloudcover = 0.2;
        const float cloudalpha = 8.0;
        const float skytint = 0.5;
        const vec3 skycolour1 = vec3(0.2, 0.4, 0.6);
        const vec3 skycolour2 = vec3(0.4, 0.7, 1.0);
        
        const mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );
        
        uniform float2 iResolution;
        uniform float2 iMouse;
        uniform float iTime;
        
        vec2 hash( vec2 p ) {
            p = vec2(dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)));
            return -1.0 + 2.0*fract(sin(p)*43758.5453123);
        }
        
        float noise( in vec2 p ) {
            const float K1 = 0.366025404; // (sqrt(3)-1)/2;
            const float K2 = 0.211324865; // (3-sqrt(3))/6;
            vec2 i = floor(p + (p.x+p.y)*K1);	
            vec2 a = p - i + (i.x+i.y)*K2;
            vec2 o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0); //vec2 of = 0.5 + 0.5*vec2(sign(a.x-a.y), sign(a.y-a.x));
            vec2 b = a - o + K2;
            vec2 c = a - 1.0 + 2.0*K2;
            vec3 h = max(0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );
            vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
            return dot(n, vec3(70.0));	
        }
        
        float fbm(vec2 n) {
            float total = 0.0, amplitude = 0.1;
            for (int i = 0; i < 7; i++) {
                total += noise(n) * amplitude;
                n = m * n;
                amplitude *= 0.4;
            }
            return total;
        }
        
        // -----------------------------------------------
        
        half4 main(in vec2 fragCoord ) {
            vec2 p = fragCoord.xy / iResolution.xy;
            vec2 uv = p*vec2(iResolution.x/iResolution.y,1.0);    
            float time = iTime * speed;
            float q = fbm(uv * cloudscale * 0.5);
            
            //ridged noise shape
            float r = 0.0;
            uv *= cloudscale;
            uv -= q - time;
            float weight = 0.8;
            for (int i=0; i<8; i++){
                r += abs(weight*noise( uv ));
                uv = m*uv + time;
                weight *= 0.7;
            }
            
            //noise shape
            float f = 0.0;
            uv = p*vec2(iResolution.x/iResolution.y,1.0);
            uv *= cloudscale;
            uv -= q - time;
            weight = 0.7;
            for (int i=0; i<8; i++){
                f += weight*noise( uv );
                uv = m*uv + time;
                weight *= 0.6;
            }
            
            f *= r + f;
            
            //noise colour
            float c = 0.0;
            time = iTime * speed * 2.0;
            uv = p*vec2(iResolution.x/iResolution.y,1.0);
            uv *= cloudscale*2.0;
            uv -= q - time;
            weight = 0.4;
            for (int i=0; i<7; i++){
                c += weight*noise( uv );
                uv = m*uv + time;
                weight *= 0.6;
            }
            
            //noise ridge colour
            float c1 = 0.0;
            time = iTime * speed * 3.0;
            uv = p*vec2(iResolution.x/iResolution.y,1.0);
            uv *= cloudscale*3.0;
            uv -= q - time;
            weight = 0.4;
            for (int i=0; i<7; i++){
                c1 += abs(weight*noise( uv ));
                uv = m*uv + time;
                weight *= 0.6;
            }
            
            c += c1;
            
            vec3 skycolour = mix(skycolour2, skycolour1, p.y);
            vec3 cloudcolour = vec3(1.1, 1.1, 0.9) * clamp((clouddark + cloudlight*c), 0.0, 1.0);
           
            f = cloudcover + cloudalpha*f*r;
            
            vec3 result = mix(skycolour, clamp(skytint * skycolour + cloudcolour, 0.0, 1.0), clamp(f + c, 0.0, 1.0));
            
            return vec4( result, 1.0 );
        }
""".trimIndent()
)

/**
 * Shadertoy's default shader
 */
val StarShader = RuntimeShader(
    """
        // Star Nest by Pablo Roman Andrioli
        
        // This content is under the MIT License.
        
        const int iterations = 17;
        const float formuparam = 0.53;
        
        const int volsteps = 20;
        const float stepsize = 0.1;
        
        const float zoom  = 0.800;
        const float tile  = 0.850;
        const float speed = 0.010 ;
        
        const float brightness = 0.0015;
        const float darkmatter = 0.300;
        const float distfading = 0.730;
        const float saturation = 0.850;
        
        // 声明内置变量
        uniform float2 resolution;
        uniform float2 iMouse;
        uniform float time;
        
        // 将返回类型从half4改为vec4
        vec4 main( in float2 fragCoord )
        {
            //get coords and direction
            vec2 uv = fragCoord.xy / resolution.xy - 0.5;
            uv.y *= resolution.y / resolution.x;
            vec3 dir = vec3(uv * zoom, 1.0);
            float time = time * speed + 0.25;
        
            //mouse rotation
            float a1 = 0.5 + iMouse.x / resolution.x * 2.0;
            float a2 = 0.8 + iMouse.y / resolution.y * 2.0;
            mat2 rot1 = mat2(cos(a1), sin(a1), -sin(a1), cos(a1));
            mat2 rot2 = mat2(cos(a2), sin(a2), -sin(a2), cos(a2));
            dir.xz *= rot1;
            dir.xy *= rot2;
            vec3 from = vec3(1.0, 0.5, 0.5);
            from += vec3(time * 2.0, time, -2.0);
            from.xz *= rot1;
            from.xy *= rot2;
        
            //volumetric rendering
            float s = 0.1, fade = 1.0;
            vec3 v = vec3(0.0);
            for (int r = 0; r < volsteps; r++) {
                vec3 p = from + s * dir * 0.5;
                p = abs(vec3(tile) - mod(p, vec3(tile * 2.0))); // tiling fold
                float pa, a = pa = 0.0;
                for (int i = 0; i < iterations; i++) { 
                    p = abs(p) / dot(p, p) - formuparam; // the magic formula
                    a += abs(length(p) - pa); // absolute sum of average change
                    pa = length(p);
                }
                float dm = max(0.0, darkmatter - a * a * 0.001); //dark matter
                a *= a * a; // add contrast
                if (r > 6) fade *= 1.0 - dm; // dark matter, don't render near
                //v+=vec3(dm,dm*.5,0.);
                v += fade;
                v += vec3(s, s * s, s * s * s * s) * a * brightness * fade; // coloring based on distance
                fade *= distfading; // distance fading
                s += stepsize;
            }
            v = mix(vec3(length(v)), v, saturation); //color adjust
            return vec4(v * 0.01, 1.0);    
        }
    """
)

/**
 * https://shaders.skia.org/?id=e0ec9ef204763445036d8a157b1b5c8929829c3e1ee0a265ed984aeddc8929e2
 */
val GradientShader = RuntimeShader(
    """
        uniform float2 resolution;
        uniform float time;
        
        vec4 main(vec2 fragCoord) {
            // Normalized pixel coordinates (from 0 to 1)
            vec2 uv = fragCoord/resolution.xy;
    
            // Time varying pixel color
            vec3 col = 0.8 + 0.2 * cos(time*2.0+uv.xxx*2.0+vec3(1,2,4));
    
            // Output to screen
            return vec4(col,1.0);
        }
    """
)

/**
 * From: https://shaders.skia.org/?id=de2a4d7d893a7251eb33129ddf9d76ea517901cec960db116a1bbd7832757c1f
 */
val NoodleZoomShader = RuntimeShader(
    """
        uniform float2 resolution;
        uniform float time;

        // Source: @notargs https://twitter.com/notargs/status/1250468645030858753
        float f(vec3 p) {
            p.z -= time * 10.;
            float a = p.z * .1;
            p.xy *= mat2(cos(a), sin(a), -sin(a), cos(a));
            return .1 - length(cos(p.xy) + sin(p.yz));
        }
        
        half4 main(vec2 fragcoord) { 
            vec3 d = .5 - fragcoord.xy1 / resolution.y;
            vec3 p=vec3(0);
            for (int i = 0; i < 32; i++) {
              p += f(p) * d;
            }
            return ((sin(p) + vec3(2, 5, 12)) / length(p)).xyz1;
        }
    """
)

/**
 * From: https://www.shadertoy.com/view/4tjSDt
 */
val WarpSpeedShader = RuntimeShader(
    """
        // 'Warp Speed 2'
        // David Hoskins 2015.
        // License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

        // Fork of:-   https://www.shadertoy.com/view/Msl3WH
        //----------------------------------------------------------------------------------------
        uniform float2 resolution;      // Viewport resolution (pixels)
        uniform float  time;            // Shader playback time (s)

        vec4 main( in float2 fragCoord )
        {
            float s = 0.0, v = 0.0;
            vec2 uv = (fragCoord / resolution.xy) * 2.0 - 1.;
            float time = (time-2.0)*58.0;
            vec3 col = vec3(0);
            vec3 init = vec3(sin(time * .0032)*.3, .35 - cos(time * .005)*.3, time * 0.002);
            for (int r = 0; r < 100; r++) 
            {
                vec3 p = init + s * vec3(uv, 0.05);
                p.z = fract(p.z);
                // Thanks to Kali's little chaotic loop...
                for (int i=0; i < 10; i++)	p = abs(p * 2.04) / dot(p, p) - .9;
                v += pow(dot(p, p), .7) * .06;
                col +=  vec3(v * 0.2+.4, 12.-s*2., .1 + v * 1.) * v * 0.00003;
                s += .025;
            }
            return vec4(clamp(col, 0.0, 1.0), 1.0);
        }
    """.trimIndent()
)


/**
 * From https://www.shadertoy.com/view/WtBXWw
 */
val LightScatteringShader = RuntimeShader(
    """
        uniform float2 resolution;      // Viewport resolution (pixels)
        uniform float  time;            // Shader playback time (s)
        uniform float2 iMouse;           // Mouse drag pos=.xy Click pos=.zw (pixels)
        
        //Based on Naty Hoffmann and Arcot J. Preetham. Rendering out-door light scattering in real time.
        //http://renderwonk.com/publications/gdm-2002/GDM_August_2002.pdf
        
        const float fov = tan(radians(60.0));
        const float cameraheight = 5e1; //50.
        const float Gamma = 2.2;
        const float Rayleigh = 1.;
        const float Mie = 1.;
        const float RayleighAtt = 1.;
        const float MieAtt = 1.2;

        float g = -0.9;
        
        vec3 _betaR = vec3(1.95e-2, 1.1e-1, 2.94e-1); 
        vec3 _betaM = vec3(4e-2, 4e-2, 4e-2);
        
        const float ts= (cameraheight / 2.5e5);
        
        vec3 Ds = normalize(vec3(0., 0., -1.)); //sun 
        
        vec3 ACESFilm( vec3 x )
        {
            float tA = 2.51;
            float tB = 0.03;
            float tC = 2.43;
            float tD = 0.59;
            float tE = 0.14;
            return clamp((x*(tA*x+tB))/(x*(tC*x+tD)+tE),0.0,1.0);
        }
        
        vec4 main(in float2 fragCoord ) {
        
            float AR = resolution.x/resolution.y;
            float M = 1.0; //canvas.innerWidth/M //canvas.innerHeight/M --res
            
            vec2 uvMouse = (iMouse.xy / resolution.xy);
            uvMouse.x *= AR;
            
            vec2 uv0 = (fragCoord.xy / resolution.xy);
            uv0 *= M;
            //uv0.x *= AR;
            
            vec2 uv = uv0 * (2.0*M) - (1.0*M);
            uv.x *=AR;
            
            if (uvMouse.y == 0.) uvMouse.y=(0.7-(0.05*fov)); //initial view 
            if (uvMouse.x == 0.) uvMouse.x=(1.0-(0.05*fov)); //initial view
            
            Ds = normalize(vec3(uvMouse.x-((0.5*AR)), uvMouse.y-0.5, (fov/-2.0)));
            
            vec3 O = vec3(0., cameraheight, 0.);
            vec3 D = normalize(vec3(uv, -(fov*M)));
        
            vec3 color = vec3(0.);
            
            if (D.y < -ts) {
                float L = - O.y / D.y;
                O = O + D * L;
                D.y = -D.y;
                D = normalize(D);
            }
            else{
                float L1 =  O.y / D.y;
                vec3 O1 = O + D * L1;
        
                vec3 D1 = vec3(1.);
                D1 = normalize(D);
            }
            
              float t = max(0.001, D.y) + max(-D.y, -0.001);
        
              // optical depth -> zenithAngle
              float sR = RayleighAtt / t ;
              float sM = MieAtt / t ;
        
              float cosine = clamp(dot(D,Ds),0.0,1.0);
              vec3 extinction = exp(-(_betaR * sR + _betaM * sM));
        
               // scattering phase
              float g2 = g * g;
              float fcos2 = cosine * cosine;
              float miePhase = Mie * pow(1. + g2 + 2. * g * cosine, -1.5) * (1. - g2) / (2. + g2);
                //g = 0;
              float rayleighPhase = Rayleigh;
        
              vec3 inScatter = (1. + fcos2) * vec3(rayleighPhase + _betaM / _betaR * miePhase);
        
              color = inScatter*(1.0-extinction); // *vec3(1.6,1.4,1.0)
        
                // sun
              color += 0.47*vec3(1.6,1.4,1.0)*pow( cosine, 350.0 ) * extinction;
              // sun haze
              color += 0.4*vec3(0.8,0.9,1.0)*pow( cosine, 2.0 )* extinction;
            
              color = ACESFilm(color);
            
              color = pow(color, vec3(Gamma));
            
              return vec4(color, 1.);
        }
    """
)
