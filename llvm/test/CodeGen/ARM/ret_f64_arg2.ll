; RUN: llc -mtriple=arm-eabi -mattr=+vfp2 %s -o /dev/null

define double @test_f64(double %a1, double %a2) {
        ret double %a2
}

