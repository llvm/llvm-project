; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep 0.0 | count 1

declare double @abs(double)

define double @test(double %X) {
  %Y = add double %X, 0.0          ;; Should be a single add x, 0.0
  %Z = add double %Y, 0.0
  ret double %Z
}

define double @test1(double %X) {
  %Y = call double @abs(double %X)
  %Z = add double %Y, 0.0
  ret double %Z
}
