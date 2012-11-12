; RUN: opt < %s -instcombine -S | grep "fcmp uno.*0.0"
; PR1111
define i1 @test(double %X) {
  %tmp = fcmp une double %X, %X
  ret i1 %tmp
}
