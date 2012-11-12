; RUN: llc < %s -march=arm -mattr=+vfp2

define i64 @test_i64_arg_split(i64 %a1, i32 %a2, i64 %a3) {
        ret i64 %a3
}

