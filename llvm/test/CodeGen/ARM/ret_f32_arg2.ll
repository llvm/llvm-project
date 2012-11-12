; RUN: llc < %s -march=arm -mattr=+vfp2

define float @test_f32(float %a1, float %a2) {
        ret float %a2
}

