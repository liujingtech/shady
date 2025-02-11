package com.goofy.goober.shady.animated

import android.graphics.RenderEffect
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.ShaderBrush
import androidx.compose.ui.graphics.asComposeRenderEffect
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.res.imageResource
import androidx.compose.ui.unit.dp
import androidx.navigation.NavGraphBuilder
import com.goofy.goober.shaders.GradientShader
import com.goofy.goober.shaders.NoodleZoomShader
import com.goofy.goober.shaders.SkyShader
import com.goofy.goober.shaders.StarryShader
import com.goofy.goober.shaders.WarpSpeedShader
import com.goofy.goober.shaders.StarShader
import com.goofy.goober.shaders.AurorasShader
import com.goofy.goober.shady.R
import com.goofy.goober.shady.ui.DestinationScreen
import com.goofy.goober.shady.ui.HomeScreens
import com.goofy.goober.shady.ui.Screen
import com.goofy.goober.shady.ui.nestedContent
import com.goofy.goober.sketch.SketchWithCache
import com.goofy.goober.style.LargeCardShape
import com.goofy.goober.style.ShadyContainer
import com.goofy.goober.style.Slider

private val Screens = listOf(
    DestinationScreen(title = "Auroras Shader") {
        AurorasShader()
    },
    DestinationScreen(title = "Starry Shader") {
        StarryShader()
    },
    DestinationScreen(title = "Sky Shader") {
        SkyShader()
    },
    DestinationScreen(title = "Star Shader") {
        StarShader()
    },
    DestinationScreen(title = "Gradient Shader") {
        GradientShader()
    },
    DestinationScreen(title = "Skia Sample Shader") {
        SkiaSampleShader()
    },
    DestinationScreen(title = "Warp Speed Shader") {
        WarpSpeedShader()
    },
)

fun NavGraphBuilder.animatedShadersGraph(onNavigate: (Screen) -> Unit) {
    nestedContent(onNavigate, screens = Screens, home = HomeScreens.AnimatedShaders)
}

@Composable
fun AurorasShader(modifier: Modifier = Modifier) {
    var iTime by remember { mutableStateOf(0.15f) }
    var iMouseX by remember { mutableStateOf(2.0f) }
    var iMouseY by remember { mutableStateOf(2.0f) }
    var iMouseZ by remember { mutableStateOf(2.0f) }
    ShadyContainer(
        modifier = modifier,
        content = {
            Image(
                modifier = Modifier
                    .fillMaxSize()
                    .onSizeChanged { size ->
                        AurorasShader.setFloatUniform(
                            /* uniformName = */ "iResolution",
                            /* value1 = */ size.width.toFloat(),
                            /* value2 = */ size.height.toFloat()
                        )
                    }
                    .graphicsLayer {
                        AurorasShader.setFloatUniform("iTime", iTime)
                        AurorasShader.setFloatUniform("iMouseX", iMouseX)
                        AurorasShader.setFloatUniform("iMouseY", iMouseY)
                        AurorasShader.setFloatUniform("iMouseZ", iMouseZ)
                        renderEffect = RenderEffect
                            .createRuntimeShaderEffect(
                                /* shader = */ AurorasShader,
                                /* uniformShaderName = */ "iImage1"
                            )
                            .asComposeRenderEffect()
                    },
                bitmap = ImageBitmap.imageResource(id = R.drawable.image),
                contentDescription = null
            )
        },
        controls = {
            Slider(
                label = "iMouseX 1 = $iMouseX",
                value = iMouseX,
                onValueChange = {
                    iMouseX = it
                },
                valueRange = 200.0f..900.0f
            )
            Spacer(modifier = Modifier.height(24.dp))
            Slider(
                label = "iMouseY 2 = $iMouseY",
                value = iMouseY,
                onValueChange = {
                    iMouseY = it
                },
                valueRange = 200.0f..900.0f
            )
            Spacer(modifier = Modifier.height(24.dp))
            Slider(
                label = "iMouseZ 3 = $iMouseZ",
                value = iMouseZ,
                onValueChange = {
                    iMouseZ = it
                },
                valueRange = 200.0f..900.0f
            )
            Spacer(modifier = Modifier.height(24.dp))
            Slider(
                label = "iTime = $iTime",
                value = iTime,
                onValueChange = {
                    iTime = it
                },
                valueRange = 0f..1000f
            )
            Spacer(modifier = Modifier.height(24.dp))
        }
    )
}

@Composable
fun StarryShader(modifier: Modifier = Modifier) {
    val brush = remember { ShaderBrush(StarryShader) }
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        SketchWithCache(
            speed = 1f,
            modifier = modifier
                .fillMaxSize(0.8f)
                .clip(LargeCardShape)
        ) { time ->
            StarryShader.setFloatUniform(
                "iResolution",
                this.size.width, this.size.height
            )
            StarryShader.setFloatUniform("iTime", time)
            StarryShader.setFloatUniform("iMouse", 500f, 500f)
            onDrawBehind {
                drawRect(brush)
            }
        }
    }
}

@Composable
fun SkyShader(modifier: Modifier = Modifier) {
    val brush = remember { ShaderBrush(SkyShader) }
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        SketchWithCache(
            speed = 1f,
            modifier = modifier
                .fillMaxSize(0.8f)
                .clip(LargeCardShape)
        ) { time ->
            SkyShader.setFloatUniform(
                "iResolution",
                this.size.width, this.size.height
            )
            SkyShader.setFloatUniform("iTime", time)
            SkyShader.setFloatUniform("iMouse", 500f, 500f)
            onDrawBehind {
                drawRect(brush)
            }
        }
    }
}

@Composable
fun StarShader(modifier: Modifier = Modifier) {
    val brush = remember { ShaderBrush(StarShader) }
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        SketchWithCache(
            speed = 1f,
            modifier = modifier
                .fillMaxSize(0.8f)
                .clip(LargeCardShape)
        ) { time ->
            StarShader.setFloatUniform(
                "resolution",
                this.size.width, this.size.height
            )
            StarShader.setFloatUniform("time", time)
            StarShader.setFloatUniform("iMouse", 500f, 500f)
            onDrawBehind {
                drawRect(brush)
            }
        }
    }
}

@Composable
fun GradientShader(modifier: Modifier = Modifier) {
    val brush = remember { ShaderBrush(GradientShader) }
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        SketchWithCache(
            speed = 1f,
            modifier = modifier
                .fillMaxSize(0.8f)
                .clip(LargeCardShape)
        ) { time ->
            GradientShader.setFloatUniform(
                "resolution",
                this.size.width, this.size.height
            )
            GradientShader.setFloatUniform("time", time)
            onDrawBehind {
                drawRect(brush)
            }
        }
    }
}

@Composable
fun SkiaSampleShader(modifier: Modifier = Modifier) {
    val brush = remember { ShaderBrush(NoodleZoomShader) }
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        SketchWithCache(
            speed = 1f,
            modifier = modifier
                .fillMaxSize(0.8f)
                .clip(LargeCardShape)
        ) { time ->
            NoodleZoomShader.setFloatUniform(
                "resolution",
                this.size.width, this.size.height
            )
            NoodleZoomShader.setFloatUniform("time", time)
            onDrawBehind {
                drawRect(brush)
            }
        }
    }
}

@Composable
fun WarpSpeedShader(modifier: Modifier = Modifier) {
    val brush = remember { ShaderBrush(WarpSpeedShader) }
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        SketchWithCache(
            speed = 1f,
            modifier = modifier
                .fillMaxSize(0.8f)
                .clip(LargeCardShape)
        ) { time ->
            WarpSpeedShader.setFloatUniform(
                "resolution",
                this.size.width, this.size.height
            )
            WarpSpeedShader.setFloatUniform("time", time)
            onDrawBehind {
                drawRect(brush)
            }
        }
    }
}
