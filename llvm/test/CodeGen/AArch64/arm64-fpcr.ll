; RUN: llc -mtriple=aarch64 < %s | FileCheck %s

define i64 @GetFpcr() {
; CHECK-LABEL: GetFpcr
; CHECK: mrs x0, FPCR
; CHECK: ret
  %1 = tail call i64 @llvm.aarch64.get.fpcr()
  ret i64 %1
}

declare i64 @llvm.aarch64.get.fpcr() #0

define i32 @GetFltRounds() {
; CHECK-LABEL: GetFltRounds
; CHECK: mrs x8, FPCR
; CHECK: add w8, w8, #1024, lsl #12
; CHECK: ubfx w0, w8, #22, #2
; CHECK: ret
  %1 = tail call i32 @llvm.flt.rounds()
  ret i32 %1
}

declare i32 @llvm.flt.rounds() #0
