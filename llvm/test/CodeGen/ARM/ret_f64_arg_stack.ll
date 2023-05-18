; RUN: llc -mtriple=arm-eabi -mattr=+vfp2 %s -o /dev/null

define double @test_double_arg_stack(i64 %a1, i32 %a2, i32 %a3, double %a4) {
        ret double %a4
}

