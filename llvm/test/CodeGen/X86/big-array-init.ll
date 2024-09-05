; RUN: llc < %s -mtriple=x86_64 | FileCheck %s

; CHECK-LABEL: bad_int:
; CHECK-NEXT: .long 1
; CHECK-NEXT: .zero 4294967292
; CHECK-NEXT: .size   bad_int, 4294967296
@bad_int = global <{ i32, [1073741823 x i32] }> <{ i32 1, [1073741823 x i32] zeroinitializer }>, align 16
