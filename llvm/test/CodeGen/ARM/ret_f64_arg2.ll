; RUN: llc < %s -march=arm -mattr=+vfp2

define double @test_f64(double %a1, double %a2) {
        ret double %a2
}

