; RUN: llc -mtriple aarch64               %s -o - | FileCheck %s
; RUN: llc -mtriple aarch64 -filetype=obj %s -o - | llvm-objdump --arch=aarch64  -d - | FileCheck %s --check-prefix=DUMP

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

define dso_local void @f(i64 %v) {
entry:
  %dummy = tail call i64 @llvm.aarch64.space(i32 32684, i64 %v)
  ret void
}
; CHECK: // SPACE 32684
; CHECK-NEXT: ret
; DUMP-LABEL: <f>:
; DUMP-NEXT: ret

declare dso_local i64 @llvm.aarch64.space(i32, i64) local_unnamed_addr #0
