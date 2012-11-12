; RUN: llc < %s -march=arm -mcpu=arm8 -mattr=+vfp2

define double @test_double_arg_reg_split(i32 %a1, double %a2) {
        ret double %a2
}

