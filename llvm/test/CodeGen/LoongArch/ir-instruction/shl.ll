; RUN: llc --mtriple=loongarch32 < %s | FileCheck %s --check-prefix=LA32
; RUN: llc --mtriple=loongarch64 < %s | FileCheck %s --check-prefix=LA64

;; Exercise the 'shl' LLVM IR: https://llvm.org/docs/LangRef.html#shl-instruction

define i1 @shl_i1(i1 %x, i1 %y) {
; LA32-LABEL: shl_i1:
; LA32:       # %bb.0:
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: shl_i1:
; LA64:       # %bb.0:
; LA64-NEXT:    jirl $zero, $ra, 0
  %shl = shl i1 %x, %y
  ret i1 %shl
}

define i8 @shl_i8(i8 %x, i8 %y) {
; LA32-LABEL: shl_i8:
; LA32:       # %bb.0:
; LA32-NEXT:    sll.w $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: shl_i8:
; LA64:       # %bb.0:
; LA64-NEXT:    sll.d $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %shl = shl i8 %x, %y
  ret i8 %shl
}

define i16 @shl_i16(i16 %x, i16 %y) {
; LA32-LABEL: shl_i16:
; LA32:       # %bb.0:
; LA32-NEXT:    sll.w $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: shl_i16:
; LA64:       # %bb.0:
; LA64-NEXT:    sll.d $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %shl = shl i16 %x, %y
  ret i16 %shl
}

define i32 @shl_i32(i32 %x, i32 %y) {
; LA32-LABEL: shl_i32:
; LA32:       # %bb.0:
; LA32-NEXT:    sll.w $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: shl_i32:
; LA64:       # %bb.0:
; LA64-NEXT:    sll.w $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %shl = shl i32 %x, %y
  ret i32 %shl
}

define i64 @shl_i64(i64 %x, i64 %y) {
; LA32-LABEL: shl_i64:
; LA32:       # %bb.0:
; LA32-NEXT:    xori $a3, $a2, 31
; LA32-NEXT:    srli.w $a4, $a0, 1
; LA32-NEXT:    srl.w $a3, $a4, $a3
; LA32-NEXT:    sll.w $a1, $a1, $a2
; LA32-NEXT:    or $a1, $a1, $a3
; LA32-NEXT:    addi.w $a3, $a2, -32
; LA32-NEXT:    slti $a4, $a3, 0
; LA32-NEXT:    maskeqz $a1, $a1, $a4
; LA32-NEXT:    sll.w $a5, $a0, $a3
; LA32-NEXT:    masknez $a4, $a5, $a4
; LA32-NEXT:    or $a1, $a1, $a4
; LA32-NEXT:    sll.w $a0, $a0, $a2
; LA32-NEXT:    srai.w $a2, $a3, 31
; LA32-NEXT:    and $a0, $a2, $a0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: shl_i64:
; LA64:       # %bb.0:
; LA64-NEXT:    sll.d $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %shl = shl i64 %x, %y
  ret i64 %shl
}

define i1 @shl_i1_3(i1 %x) {
; LA32-LABEL: shl_i1_3:
; LA32:       # %bb.0:
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: shl_i1_3:
; LA64:       # %bb.0:
; LA64-NEXT:    jirl $zero, $ra, 0
  %shl = shl i1 %x, 3
  ret i1 %shl
}

define i8 @shl_i8_3(i8 %x) {
; LA32-LABEL: shl_i8_3:
; LA32:       # %bb.0:
; LA32-NEXT:    slli.w $a0, $a0, 3
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: shl_i8_3:
; LA64:       # %bb.0:
; LA64-NEXT:    slli.d $a0, $a0, 3
; LA64-NEXT:    jirl $zero, $ra, 0
  %shl = shl i8 %x, 3
  ret i8 %shl
}

define i16 @shl_i16_3(i16 %x) {
; LA32-LABEL: shl_i16_3:
; LA32:       # %bb.0:
; LA32-NEXT:    slli.w $a0, $a0, 3
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: shl_i16_3:
; LA64:       # %bb.0:
; LA64-NEXT:    slli.d $a0, $a0, 3
; LA64-NEXT:    jirl $zero, $ra, 0
  %shl = shl i16 %x, 3
  ret i16 %shl
}

define i32 @shl_i32_3(i32 %x) {
; LA32-LABEL: shl_i32_3:
; LA32:       # %bb.0:
; LA32-NEXT:    slli.w $a0, $a0, 3
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: shl_i32_3:
; LA64:       # %bb.0:
; LA64-NEXT:    slli.d $a0, $a0, 3
; LA64-NEXT:    jirl $zero, $ra, 0
  %shl = shl i32 %x, 3
  ret i32 %shl
}

define i64 @shl_i64_3(i64 %x) {
; LA32-LABEL: shl_i64_3:
; LA32:       # %bb.0:
; LA32-NEXT:    slli.w $a1, $a1, 3
; LA32-NEXT:    srli.w $a2, $a0, 29
; LA32-NEXT:    or $a1, $a1, $a2
; LA32-NEXT:    slli.w $a0, $a0, 3
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: shl_i64_3:
; LA64:       # %bb.0:
; LA64-NEXT:    slli.d $a0, $a0, 3
; LA64-NEXT:    jirl $zero, $ra, 0
  %shl = shl i64 %x, 3
  ret i64 %shl
}
