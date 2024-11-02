; RUN: opt -S -passes=callsite-splitting -callsite-splitting-duplication-threshold=100000000 < %s | FileCheck -enable-var-scope %s

; Convergent calls should not be duplicated in this case
; CHECK-LABEL: define void @convergent_caller(
; CHECK: call void @convergent_callee(
; CHECK-NOT: call void @convergent_callee(
define void @convergent_caller(i1 %c, i8* %a_elt, i8* %b_elt) #0 {
entry:
  br label %Top

Top:
  %tobool1 = icmp eq i8* %a_elt, null
  br i1 %tobool1, label %CallSiteBB, label %NextCond

NextCond:
  %cmp = icmp ne i8* %b_elt, null
  br i1 %cmp, label %CallSiteBB, label %End

CallSiteBB:
  %p = phi i1 [ false, %Top ], [ %c, %NextCond ]
  call void @convergent_callee(i8* %a_elt, i1 %p)
  br label %End

End:
  ret void
}

; CHECK-LABEL: define void @convergent_callee(
; CHECK: call void @convergent_external(
; CHECK-NOT: call void @convergent_external(
define void @convergent_callee(i8* %a_elt, i1 %c) #0 {
entry:
  %tobool = icmp ne i8* %a_elt, null
  br i1 %tobool, label %then, label %endif

then:
  br label %endif

endif:
  call void @convergent_external(i8* %a_elt) #0
  ret void
}

; Make sure an otherwise identical function is transformed
; CHECK-LABEL: define void @reference_caller(
; CHECK: call void @nonconvergent_callee(
; CHECK: call void @nonconvergent_callee(
define void @reference_caller(i1 %c, i8* %a_elt, i8* %b_elt) #1 {
entry:
  br label %Top

Top:
  %tobool1 = icmp eq i8* %a_elt, null
  br i1 %tobool1, label %CallSiteBB, label %NextCond

NextCond:
  %cmp = icmp ne i8* %b_elt, null
  br i1 %cmp, label %CallSiteBB, label %End

CallSiteBB:
  %p = phi i1 [ false, %Top ], [ %c, %NextCond ]
  call void @nonconvergent_callee(i8* %a_elt, i1 %p)
  br label %End

End:
  ret void
}

; CHECK-LABEL: define void @nonconvergent_callee(
; CHECK: call void @nonconvergent_external(
; CHECK-NOT: call void @nonconvergent_external(
define void @nonconvergent_callee(i8* %a_elt, i1 %c) #1 {
entry:
  %tobool = icmp ne i8* %a_elt, null
  br i1 %tobool, label %then, label %endif

then:
  br label %endif

endif:
  call void @nonconvergent_external(i8* %a_elt)
  ret void
}

declare void @convergent_external(i8*) #0
declare void @nonconvergent_external(i8*) #1

attributes #0 = { convergent nounwind }
attributes #1 = { nounwind }
