; RUN: llc < %s -march=arm -mattr=+vfp2

define i64 @test_i64(i64 %a1, i64 %a2) {
        ret i64 %a2
}

