; Check FP options -fno-trapping-math and -fdenormal-fp-math. They are passed as
; function attributes, which map on to build attributes ABI_FP_exceptions ABI_FP_denormal.
; In the backend we have a check to see if all functions have consistent function
; attributes values. This checks the "default" behaviour when these FP function
; attributes are not set at all.

; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15  | FileCheck %s

; CHECK: .eabi_attribute 20, 1
; CHECK: .eabi_attribute 21, 1

define i32 @foo_no_fn_attr() local_unnamed_addr #0 {
entry:
  ret i32 42
}

attributes #0 = { minsize norecurse nounwind optsize readnone }
