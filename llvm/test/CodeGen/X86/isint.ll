; RUN: llc < %s -mtriple=x86_64-pc-unknown -mattr=+sse2 -mcpu=penryn | FileCheck %s
; RUN: llc < %s -mtriple=i686-pc-unknown -mattr=+sse2 -mcpu=penryn | FileCheck %s

; PR19059
; RUN: llc < %s -mtriple=i686-pc-unknown -mattr=+sse2 -mcpu=penryn | FileCheck -check-prefix=CHECK32 %s

define i32 @isint_return(double %d) nounwind {
; CHECK-LABEL: isint_return:
; CHECK-NOT: xor
; CHECK: cvt
  %i = fptosi double %d to i32
; CHECK-NEXT: cvt
  %e = sitofp i32 %i to double
; CHECK: cmpeqsd
  %c = fcmp oeq double %d, %e
; CHECK32-NOT: movd {{.*}}, %r{{.*}}
; CHECK32-NOT: andq
; CHECK-NEXT: movd
; CHECK-NEXT: andl
  %z = zext i1 %c to i32
  ret i32 %z
}

define i32 @isint_float_return(float %f) nounwind {
; CHECK-LABEL: isint_float_return:
; CHECK-NOT: xor
; CHECK: cvt
  %i = fptosi float %f to i32
; CHECK-NEXT: cvt
  %g = sitofp i32 %i to float
; CHECK: cmpeqss
  %c = fcmp oeq float %f, %g
; CHECK-NOT: movd {{.*}}, %r{{.*}}
; CHECK-NEXT: movd
; CHECK-NEXT: andl
  %z = zext i1 %c to i32
  ret i32 %z
}

declare void @foo()

define void @isint_branch(double %d) nounwind {
; CHECK-LABEL: isint_branch:
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
