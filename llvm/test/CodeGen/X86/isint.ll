; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 > %t
; RUN: not grep cmp %t
; RUN: not grep xor %t
; RUN: grep jne %t | count 1
; RUN: grep jp %t | count 1
; RUN: grep setnp %t | count 1
; RUN: grep sete %t | count 1
; RUN: grep and %t | count 1
; RUN: grep cvt %t | count 4

define i32 @isint_return(double %d) nounwind {
  %i = fptosi double %d to i32
  %e = sitofp i32 %i to double
  %c = fcmp oeq double %d, %e
  %z = zext i1 %c to i32
  ret i32 %z
}

declare void @foo()

define void @isint_branch(double %d) nounwind {
  %i = fptosi double %d to i32
  %e = sitofp i32 %i to double
  %c = fcmp oeq double %d, %e
  br i1 %c, label %true, label %false
true:
  call void @foo()
  ret void
false:
  ret void
}
