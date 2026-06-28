; Check for functions that are already generated (incorrectly), functions that need
; generation, and functions where the prefixes mismatch on the constant pool

; RUN: llc -mtriple=riscv32 -mattr=+v -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,RV32
; RUN: llc -mtriple=riscv64 -mattr=+v -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,RV64

define <8 x i64> @needs_generated(<8 x i64> %x) {
  %s = shufflevector <8 x i64> poison, <8 x i64> %x, <8 x i32> <i32 9, i32 10, i32 8, i32 9, i32 15, i32 8, i32 8, i32 11>
  ret <8 x i64> %s
}

define <8 x i64> @only_one_prefix(<8 x i64> %x, <8 x i64> %y) {
  %s = shufflevector <8 x i64> %x, <8 x i64> %y, <8 x i32> <i32 1, i32 2, i32 10, i32 5, i32 1, i32 10, i32 3, i32 13>
  ret <8 x i64> %s
}

define <8 x i64> @differing_prefixes(<8 x i64> %x) {
  %s = shufflevector <8 x i64> %x, <8 x i64> <i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5, i64 5>, <8 x i32> <i32 0, i32 3, i32 10, i32 9, i32 4, i32 1, i32 7, i32 14>
  ret <8 x i64> %s
}


define <8 x i8> @regenerate_incorrect(<8 x i8> %v, <8 x i8> %w) {
; CHECK-LABEL: LCPI26_0
; CHECK-NEXT: 	.byte	3                               # 0x3
; CHECK-NEXT: 	.byte	0                               # 0x0
; CHECK-NEXT: 	.byte	4                               # 0x4
; CHECK-NEXT: 	.byte	2                               # 0x2
; CHECK-NEXT: 	.byte	2                               # 0x2
; CHECK-NEXT: 	.byte	6                               # 0x6
; CHECK-NEXT: 	.byte	5                               # 0x5
; CHECK-NEXT: 	.byte	2                               # 0x2
; CHECK-LABEL: regenerate_incorrect:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lui a0, %hi(.LCPI26_0)
; CHECK-NEXT:    addi a0, a0, %lo(.LCPI26_0)
; CHECK-NEXT:    vsetivli zero, 8, e8, mf2, ta, ma
; CHECK-NEXT:    vle8.v v10, (a0)
; CHECK-NEXT:    li a0, 20
; CHECK-NEXT:    vmv.s.x v0, a0
; CHECK-NEXT:    vmerge.vvm v9, v9, v8, v0
; CHECK-NEXT:    vrgather.vv v8, v9, v10
; CHECK-NEXT:    ret
  %shuff = shufflevector <8 x i8> %v, <8 x i8> %w, <8 x i32> <i32 2, i32 8, i32 4, i32 2, i32 2, i32 14, i32 8, i32 2>
  ret <8 x i8> %shuff
}
