; RUN: llc -mtriple=arm-eabi -mattr=+vfp2 %s -o /dev/null

define float @test_f32(float %a1, float %a2) {
        ret float %a2
}

