; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-linux-gnueabi | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios | FileCheck %s --check-prefix=THUMB

; Fast-isel can't handle non-double multi-reg retvals.
; This test just check to make sure we don't hit the assert in FinishCall.
define <16 x i8> @foo() nounwind ssp {
entry:
  ret <16 x i8> zeroinitializer
}

define void @t1() nounwind ssp {
entry:
; ARM: @t1
; THUMB: @t1
  %call = call <16 x i8> @foo()
  ret void
}
