; RUN: llc -fast-isel -o - %s -fast-isel-report-on-fallback 2>&1 | FileCheck %s
; Make sure FastISel report a warming when we asked it to do so.
; Note: This test needs to use whatever is not supported by FastISel.
;       Thus, this test may fail because inline asm gets supported in FastISel.
;       To fix this, use something else that's not supported (e.g., weird types).
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

; CHECK: warning: Instruction selection used fallback path for foo
define void @foo(){
entry:
  call void asm sideeffect "nop", "~{dirflag},~{fpsr},~{flags}"()
  ret void
}
