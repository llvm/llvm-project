; Check that calls are not inlined if the call penalty is low. The value of the
; call penalty is provided with the '--inline-call-penalty' option.
;
; RUN: opt < %s -inline --inline-call-penalty=0 --inline-threshold=5 -S | FileCheck %s
; RUN: opt < %s -inline --inline-threshold=5 -S | FileCheck %s -check-prefix=DEFAULT_CALL_PENALTY

define i32 @X9(i32 %x) nounwind {
  %x2 = add i32 %x, %x
  %x3 = add i32 %x2, %x
  %x4 = add i32 %x3, %x
  %x5 = add i32 %x4, %x
  %x6 = add i32 %x5, %x
  %x7 = add i32 %x6, %x
  %x8 = add i32 %x7, %x
  %x9 = add i32 %x8, %x

  ret i32 %x9
}

define i32 @f1(i32 %x) nounwind {
  %res = call i32 @X9(i32 %x)
  ret i32 %res
; CHECK-LABEL: @f1(
; CHECK: %res = call i32 @X9

; DEFAULT_CALL_PENALTY-LABEL: @f1(
; DEFAULT_CALL_PENALTY-NOT: call
}
