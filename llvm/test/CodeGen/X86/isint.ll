; RUN: llc < %s -march=x86 -mattr=+sse2 | FileCheck %s

define i32 @isint_return(double %d) nounwind {
; CHECK-NOT: xor
; CHECK: cvt
  %i = fptosi double %d to i32
; CHECK-NEXT: cvt
  %e = sitofp i32 %i to double
; CHECK: cmpeqsd
  %c = fcmp oeq double %d, %e
; CHECK-NEXT: movd
; CHECK-NEXT: andl
  %z = zext i1 %c to i32
  ret i32 %z
}

declare void @foo()

define void @isint_branch(double %d) nounwind {
; CHECK: cvt
  %i = fptosi double %d to i32
; CHECK-NEXT: cvt
  %e = sitofp i32 %i to double
; CHECK: ucomisd
  %c = fcmp oeq double %d, %e
; CHECK-NEXT: jne
; CHECK-NEXT: jp
  br i1 %c, label %true, label %false
true:
  call void @foo()
  ret void
false:
  ret void
}
