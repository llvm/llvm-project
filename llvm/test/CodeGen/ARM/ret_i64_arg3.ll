; RUN: llc < %s -march=arm -mattr=+vfp2

define i64 @test_i64_arg3(i64 %a1, i64 %a2, i64 %a3) {
        ret i64 %a3
}

