; RUN: llc -mtriple=hexagon --verify-machineinstrs < %s | FileCheck %s

; Generate code that is guaranteed to crash.  The trap is the 32-bit word
; 0x9b810001 which encodes R1 = memw(R1++#0) -- both the load destination
; and the post-increment destination write R1, triggering a hardware
; "multiple writes to register" exception.
; CHECK-LABEL: f0
; CHECK: .word 2608922625

target triple = "hexagon"

define i32 @f0() noreturn nounwind  {
entry:
  tail call void @llvm.trap()
  unreachable
}

; CHECK-LABEL: f1
; CHECK: trap0(#219)
define i32 @f1() noreturn nounwind {
entry:
  tail call void @llvm.debugtrap()
  unreachable
}

declare void @llvm.trap() nounwind
declare void @llvm.debugtrap() nounwind
