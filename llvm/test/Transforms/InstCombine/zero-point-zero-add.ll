; RUN: opt < %s -instcombine -S | grep 0.0 | count 1

declare double @abs(double)

define double @test(double %X) {
  %Y = fadd double %X, 0.0          ;; Should be a single add x, 0.0
  %Z = fadd double %Y, 0.0
  ret double %Z
}

define double @test1(double %X) {
  %Y = call double @abs(double %X)
  %Z = fadd double %Y, 0.0
  ret double %Z
}
