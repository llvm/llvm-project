; Check FP options -fno-trapping-math and -fdenormal-fp-math. They are passed
; as function attributes, which map on to build attributes ABI_FP_exceptions
; ABI_FP_denormal. In the backend we therefore have a check to see if all
; functions have consistent function attributes values. Here we check two
; functions have inconsistent values, and that a default is returned.

; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15  | FileCheck %s
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15 -enable-ipra | FileCheck %s

; CHECK: .eabi_attribute 20, 1

define i32 @foo1() local_unnamed_addr #0 {
entry:
  ret i32 42
}

define i32 @foo2() local_unnamed_addr #1 {
entry:
  ret i32 42
}

attributes #0 = { minsize norecurse nounwind optsize readnone "denormal-fp-math"="preserve-sign,preserve-sign"}
attributes #1 = { minsize norecurse nounwind optsize readnone "denormal-fp-math"="positive-zero,positive-zero"}
