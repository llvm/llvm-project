; RUN: llc -mtriple=hexagon --verify-machineinstrs < %s | FileCheck %s

; Generate code that is guaranteed to crash. At the moment, it's a
; misaligned load.
; CHECK-LABEL: f0
; CHECK: memd(##3134984174)

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
