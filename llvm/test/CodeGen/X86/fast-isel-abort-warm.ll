; RUN: llc -fast-isel -o - %s -fast-isel-report-on-fallback -pass-remarks-missed=isel 2>&1 | FileCheck %s
; Make sure FastISel report a warming when we asked it to do so.
; Note: This test needs to use whatever is not supported by FastISel.
;       Thus, this test may fail because inline asm gets supported in FastISel.
;       To fix this, use something else that's not supported (e.g., weird types).
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

; CHECK: remark: <unknown>:0:0: FastISel missed call:   call void asm sideeffect
; CHECK: warning: Instruction selection used fallback path for foo
define void @foo(){
entry:
  call void asm sideeffect "nop", "~{dirflag},~{fpsr},~{flags}"()
  ret void
}

; CHECK: remark: <unknown>:0:0: FastISel missed:   store i128
; CHECK: warning: Instruction selection used fallback path for test_instruction_fallback
define void @test_instruction_fallback(i128* %ptr){
  %v1 = load i128, i128* %ptr
  %result = add i128 %v1, %v1
  store i128 %result, i128 * %ptr
  ret void
}

; CHECK-NOT: remark: <unknown>:0:0: FastISel missed
; CHECK-NOT: warning: Instruction selection used fallback path for test_instruction_not_fallback
define i32 @test_instruction_not_fallback(i32 %a){
  %result = add i32 %a, %a
  ret i32 %result
}
