; Check that calls are not inlined if the call penalty is low. The value of the
; call penalty is provided with the '--inline-call-penalty' option.
;
; RUN: opt < %s -passes=inline --inline-call-penalty=0 --inline-threshold=5 -S | FileCheck %s
; RUN: opt < %s -passes=inline --inline-threshold=5 -S | FileCheck %s -check-prefix=DEFAULT_CALL_PENALTY

declare void @extern()

define void @X9() nounwind {
  call void @extern() "call-inline-cost"="30"
  ret void
}

define void @f1() nounwind {
  call void @X9()
  ret void
; CHECK-LABEL: @f1(
; CHECK: call void @X9

; DEFAULT_CALL_PENALTY-LABEL: @f1(
; DEFAULT_CALL_PENALTY: call void @extern
; DEFAULT_CALL_PENALTY-NOT: call void @X9
}
