; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {fcmp uno.*0.0}
; PR1111
define i1 @test(double %X) {
  %tmp = fcmp une double %X, %X
  ret i1 %tmp
}
