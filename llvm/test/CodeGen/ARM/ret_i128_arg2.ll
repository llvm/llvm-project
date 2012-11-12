; RUN: llc < %s -march=arm -mattr=+vfp2

define i128 @test_i128(i128 %a1, i128 %a2, i128 %a3) {
        ret i128 %a3
}

