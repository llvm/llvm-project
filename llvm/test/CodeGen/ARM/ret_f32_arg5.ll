; RUN: llc -mtriple=arm-eabi -mattr=+vfp2 %s -o /dev/null

define float @test_f32_arg5(float %a1, float %a2, float %a3, float %a4, float %a5) {
        ret float %a5
}

