; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -verify-machineinstrs -O0 < %s

define double @test() {
  ret double 1.000000e+00
}

@g = common global double 0.000000e+00, align 8

define double @testitd() {
  %g = load double, double* @g, align 8
  ret double %g
}

