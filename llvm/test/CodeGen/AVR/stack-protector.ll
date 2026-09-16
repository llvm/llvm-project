; RUN: llc -mtriple=avr -verify-machineinstrs < %s | FileCheck %s

; The stack protector libcalls must be available on AVR. Lowering `sspreq`
; needs both `__stack_chk_guard`, to load the canary, and `__stack_chk_fail`,
; for the failure path. When either is missing from AVR's
; SystemRuntimeLibrary, SelectionDAG reports "unable to lower stackguard" and
; then dies in TargetLowering::makeLibCall with "unsupported library call
; operation".

define void @func() sspreq nounwind {
; CHECK-LABEL: func:
; CHECK: __stack_chk_guard
; CHECK: __stack_chk_fail
  %alloca = alloca i32, align 4
  call void @capture(ptr %alloca)
  ret void
}

declare void @capture(ptr)
