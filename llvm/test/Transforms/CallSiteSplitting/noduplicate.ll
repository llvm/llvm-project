; RUN: opt -S -passes=callsite-splitting -callsite-splitting-duplication-threshold=100000000 < %s | FileCheck -enable-var-scope %s

; Noduplicate calls should not be duplicated
; CHECK-LABEL: define void @noduplicate_caller(
; CHECK: call void @noduplicate_callee(
; CHECK-NOT: call void @noduplicate_callee(
define void @noduplicate_caller(i1 %c, i8* %a_elt, i8* %b_elt) #0 {
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
  call void @noduplicate_callee(i8* %a_elt, i1 %p)
  br label %End

End:
  ret void
}

; CHECK-LABEL: define void @noduplicate_callee(
; CHECK: call void @noduplicate_external(
; CHECK-NOT: call void @noduplicate_external(
define void @noduplicate_callee(i8* %a_elt, i1 %c) #0 {
entry:
  %tobool = icmp ne i8* %a_elt, null
  br i1 %tobool, label %then, label %endif

then:
  br label %endif

endif:
  call void @noduplicate_external(i8* %a_elt) #0
  ret void
}

; Make sure an otherwise identical function is transformed
; CHECK-LABEL: define void @reference_caller(
; CHECK: call void @nonnoduplicate_callee(
; CHECK: call void @nonnoduplicate_callee(
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
  call void @nonnoduplicate_callee(i8* %a_elt, i1 %p)
  br label %End

End:
  ret void
}

; CHECK-LABEL: define void @nonnoduplicate_callee(
; CHECK: call void @nonnoduplicate_external(
; CHECK-NOT: call void @nonnoduplicate_external(
define void @nonnoduplicate_callee(i8* %a_elt, i1 %c) #1 {
entry:
  %tobool = icmp ne i8* %a_elt, null
  br i1 %tobool, label %then, label %endif

then:
  br label %endif

endif:
  call void @nonnoduplicate_external(i8* %a_elt)
  ret void
}

declare void @noduplicate_external(i8*) #0
declare void @nonnoduplicate_external(i8*) #1

attributes #0 = { noduplicate nounwind }
attributes #1 = { nounwind }

