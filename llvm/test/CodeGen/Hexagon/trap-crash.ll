; RUN: llc -mtriple=hexagon --verify-machineinstrs < %s | FileCheck %s

; Generate code that is guaranteed to crash.  The trap is a 32-bit zero word
; that decodes as a duplex writing R0 from both slots, which triggers a
; hardware exception.
; CHECK-LABEL: f0
; CHECK: .word 0

target triple = "hexagon"

define i32 @f0() noreturn nounwind  {
entry:
  tail call void @llvm.trap()
  unreachable
}

; CHECK-LABEL: f1
; CHECK: brkpt
define i32 @f1() noreturn nounwind {
entry:
  tail call void @llvm.debugtrap()
  unreachable
}

declare void @llvm.trap() nounwind
declare void @llvm.debugtrap() nounwind
