; RUN: llc -mtriple armv8-eabi -mcpu=cortex-a57 -o - %s | FileCheck %s
; RUN: llc -mtriple thumbv8-eabi -mcpu=cortex-a57 -o - %s | FileCheck %s

define void @hint_dbg() {
entry:
  call void @llvm.arm.dbg(i32 0)
  ret void
}

declare void @llvm.arm.dbg(i32)

; CHECK: dbg #0

