; RUN: llc -mtriple=x86_64-apple-darwin -mcpu=znver1 -frame-pointer=all -filetype=obj %s -o - | llvm-objdump -d - | FileCheck %s
; RUN: llc -mtriple=x86_64-apple-darwin -mcpu=znver1 -frame-pointer=all -filetype=asm %s -o - | llvm-mc -triple=x86_64-apple-darwin -mcpu=znver1 -filetype=obj - | llvm-objdump -d - | FileCheck %s

define void @nop_test() {
entry:
; CHECK:  0: 55
; CHECK:  1: 48 89 e5

; CHECK:  4: 90
; CHECK:  5: 66 90
; CHECK:  7: 0f 1f 00
; CHECK:  a: 0f 1f 40 08
; CHECK:  e: 0f 1f 44 00 08
; CHECK: 13: 66 0f 1f 44 00 08
; CHECK: 19: 0f 1f 80 00 02 00 00
; CHECK: 20: 0f 1f 84 00 00 02 00 00
; CHECK: 28: 66 0f 1f 84 00 00 02 00 00
; CHECK: 31: 2e 66 0f 1f 84 00 00 02 00 00
; CHECK: 3b: 66 2e 66 0f 1f 84 00 00 02 00 00
; CHECK: 46: 66 66 2e 66 0f 1f 84 00 00 02 00 00
; CHECK: 52: 66 66 66 2e 66 0f 1f 84 00 00 02 00 00
; CHECK: 5f: 66 66 66 66 2e 66 0f 1f 84 00 00 02 00 00
; CHECK: 6d: 66 66 66 66 66 2e 66 0f 1f 84 00 00 02 00 00

; CHECK: 7c: 5d
; CHECK: 7d: c3

  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64  0, i32  0)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64  1, i32  1)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64  2, i32  2)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64  3, i32  3)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64  4, i32  4)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64  5, i32  5)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64  6, i32  6)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64  7, i32  7)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64  8, i32  8)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64  9, i32  9)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 10, i32 10)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 11, i32 11)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 12, i32 12)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 13, i32 13)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 14, i32 14)
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 15, i32 15)
; Add an extra stackmap with a zero-length shadow to thwart the shadow
; optimization. This will force all 15 bytes of the previous shadow to be
; padded with nops.
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 16, i32 0)
  ret void
}

declare void @llvm.experimental.stackmap(i64, i32, ...)
