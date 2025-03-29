; RUN: llc < %s -mtriple=mipsel-unknown-linux-gnu | FileCheck %s --check-prefixes=MIPS32
; RUN: llc < %s -mtriple=mips64el-unknown-linux-gnuabi64 | FileCheck %s --check-prefixes=MIPS64
; RUN: llc < %s -mtriple=mips64el-unknown-linux-gnuabi64 | FileCheck %s --check-prefixes=MIPS64

define i32 @shl_32(i32 %a, i32 %b) {
; MIPS32-LABLE:   shl_32:
; MIPS32:	  # %bb.0:
; MIPS32-NEXT:    jr	$ra
; MIPS32-NEXT:    sllv	$2, $4, $5
; MIPS64-LABLE:   shl_32:
; MIPS64:	  # %bb.0:
; MIPS64-NEXT:    sll   $1, $5, 0
; MIPS64-NEXT:    sll   $2, $4, 0
; MIPS64-NEXT:    jr	$ra
; MIPS64-NEXT:    sllv	$2, $2, $1
  %_1 = and i32 %b, 31
  %_0 = shl i32 %a, %_1
  ret i32 %_0
}

define i32 @lshr_32(i32 %a, i32 %b) {
; MIPS32-LABLE:   lshr_32:
; MIPS32:	  # %bb.0:
; MIPS32-NEXT:    jr	$ra
; MIPS32-NEXT:    srlv	$2, $4, $5
; MIPS64-LABLE:   lshr_32:
; MIPS64:	  # %bb.0:
; MIPS64-NEXT:    sll   $1, $5, 0
; MIPS64-NEXT:    sll   $2, $4, 0
; MIPS64-NEXT:    jr	$ra
; MIPS64-NEXT:    srlv	$2, $2, $1
  %_1 = and i32 %b, 31
  %_0 = lshr i32 %a, %_1
  ret i32 %_0
}

define i32 @ashr_32(i32 %a, i32 %b) {
; MIPS32-LABLE:   ashr_32:
; MIPS32:	  # %bb.0:
; MIPS32-NEXT:    jr	$ra
; MIPS32-NEXT:    srav	$2, $4, $5
; MIPS64-LABLE:   ashr_32:
; MIPS64:	  # %bb.0:
; MIPS64-NEXT:    sll   $1, $5, 0
; MIPS64-NEXT:    sll   $2, $4, 0
; MIPS64-NEXT:    jr	$ra
; MIPS64-NEXT:    srav	$2, $2, $1
  %_1 = and i32 %b, 31
  %_0 = ashr i32 %a, %_1
  ret i32 %_0
}

define i64 @shl_64(i64 %a, i64 %b) {
; MIPS64-LABLE:   shl_64:
; MIPS64:	  # %bb.0:
; MIPS64-NEXT:    sll   $1, $5, 0
; MIPS64-NEXT:    jr	$ra
; MIPS64-NEXT:    dsllv	$2, $4, $1
  %_1 = and i64 %b, 63
  %_0 = shl i64 %a, %_1
  ret i64 %_0
}

define i64 @lshr_64(i64 %a, i64 %b) {
; MIPS64-LABLE:   lshr_64:
; MIPS64:	  # %bb.0:
; MIPS64-NEXT:    sll   $1, $5, 0
; MIPS64-NEXT:    jr	$ra
; MIPS64-NEXT:    dsrlv	$2, $4, $1
  %_1 = and i64 %b, 63
  %_0 = lshr i64 %a, %_1
  ret i64 %_0
}

define i64 @ashr_64(i64 %a, i64 %b) {
; MIPS64-LABLE:   ashr_64:
; MIPS64:	  # %bb.0:
; MIPS64-NEXT:    sll   $1, $5, 0
; MIPS64-NEXT:    jr	$ra
; MIPS64-NEXT:    dsrav	$2, $4, $1
  %_1 = and i64 %b, 63
  %_0 = ashr i64 %a, %_1
  ret i64 %_0
}
