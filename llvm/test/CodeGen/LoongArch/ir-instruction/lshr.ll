; RUN: llc --mtriple=loongarch32 < %s | FileCheck %s --check-prefix=LA32
; RUN: llc --mtriple=loongarch64 < %s | FileCheck %s --check-prefix=LA64

;; Exercise the 'lshr' LLVM IR: https://llvm.org/docs/LangRef.html#lshr-instruction

define i1 @lshr_i1(i1 %x, i1 %y) {
; LA32-LABEL: lshr_i1:
; LA32:       # %bb.0:
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: lshr_i1:
; LA64:       # %bb.0:
; LA64-NEXT:    jirl $zero, $ra, 0
  %lshr = lshr i1 %x, %y
  ret i1 %lshr
}

define i8 @lshr_i8(i8 %x, i8 %y) {
; LA32-LABEL: lshr_i8:
; LA32:       # %bb.0:
; LA32-NEXT:    andi $a0, $a0, 255
; LA32-NEXT:    srl.w $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: lshr_i8:
; LA64:       # %bb.0:
; LA64-NEXT:    andi $a0, $a0, 255
; LA64-NEXT:    srl.d $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %lshr = lshr i8 %x, %y
  ret i8 %lshr
}

define i16 @lshr_i16(i16 %x, i16 %y) {
; LA32-LABEL: lshr_i16:
; LA32:       # %bb.0:
; LA32-NEXT:    lu12i.w $a2, 15
; LA32-NEXT:    ori $a2, $a2, 4095
; LA32-NEXT:    and $a0, $a0, $a2
; LA32-NEXT:    srl.w $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: lshr_i16:
; LA64:       # %bb.0:
; LA64-NEXT:    lu12i.w $a2, 15
; LA64-NEXT:    ori $a2, $a2, 4095
; LA64-NEXT:    and $a0, $a0, $a2
; LA64-NEXT:    srl.d $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %lshr = lshr i16 %x, %y
  ret i16 %lshr
}

define i32 @lshr_i32(i32 %x, i32 %y) {
; LA32-LABEL: lshr_i32:
; LA32:       # %bb.0:
; LA32-NEXT:    srl.w $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: lshr_i32:
; LA64:       # %bb.0:
; LA64-NEXT:    srl.w $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %lshr = lshr i32 %x, %y
  ret i32 %lshr
}

define i64 @lshr_i64(i64 %x, i64 %y) {
; LA32-LABEL: lshr_i64:
; LA32:       # %bb.0:
; LA32-NEXT:    xori $a3, $a2, 31
; LA32-NEXT:    slli.w $a4, $a1, 1
; LA32-NEXT:    sll.w $a3, $a4, $a3
; LA32-NEXT:    srl.w $a0, $a0, $a2
; LA32-NEXT:    or $a0, $a0, $a3
; LA32-NEXT:    addi.w $a3, $a2, -32
; LA32-NEXT:    slti $a4, $a3, 0
; LA32-NEXT:    maskeqz $a0, $a0, $a4
; LA32-NEXT:    srl.w $a5, $a1, $a3
; LA32-NEXT:    masknez $a4, $a5, $a4
; LA32-NEXT:    or $a0, $a0, $a4
; LA32-NEXT:    srl.w $a1, $a1, $a2
; LA32-NEXT:    srai.w $a2, $a3, 31
; LA32-NEXT:    and $a1, $a2, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: lshr_i64:
; LA64:       # %bb.0:
; LA64-NEXT:    srl.d $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
  %lshr = lshr i64 %x, %y
  ret i64 %lshr
}

define i1 @lshr_i1_3(i1 %x) {
; LA32-LABEL: lshr_i1_3:
; LA32:       # %bb.0:
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: lshr_i1_3:
; LA64:       # %bb.0:
; LA64-NEXT:    jirl $zero, $ra, 0
  %lshr = lshr i1 %x, 3
  ret i1 %lshr
}

define i8 @lshr_i8_3(i8 %x) {
; LA32-LABEL: lshr_i8_3:
; LA32:       # %bb.0:
; LA32-NEXT:    andi $a0, $a0, 248
; LA32-NEXT:    srli.w $a0, $a0, 3
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: lshr_i8_3:
; LA64:       # %bb.0:
; LA64-NEXT:    andi $a0, $a0, 248
; LA64-NEXT:    srli.d $a0, $a0, 3
; LA64-NEXT:    jirl $zero, $ra, 0
  %lshr = lshr i8 %x, 3
  ret i8 %lshr
}

define i16 @lshr_i16_3(i16 %x) {
; LA32-LABEL: lshr_i16_3:
; LA32:       # %bb.0:
; LA32-NEXT:    lu12i.w $a1, 15
; LA32-NEXT:    ori $a1, $a1, 4088
; LA32-NEXT:    and $a0, $a0, $a1
; LA32-NEXT:    srli.w $a0, $a0, 3
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: lshr_i16_3:
; LA64:       # %bb.0:
; LA64-NEXT:    lu12i.w $a1, 15
; LA64-NEXT:    ori $a1, $a1, 4088
; LA64-NEXT:    and $a0, $a0, $a1
; LA64-NEXT:    srli.d $a0, $a0, 3
; LA64-NEXT:    jirl $zero, $ra, 0
  %lshr = lshr i16 %x, 3
  ret i16 %lshr
}

define i32 @lshr_i32_3(i32 %x) {
; LA32-LABEL: lshr_i32_3:
; LA32:       # %bb.0:
; LA32-NEXT:    srli.w $a0, $a0, 3
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: lshr_i32_3:
; LA64:       # %bb.0:
; LA64-NEXT:    addi.w $a1, $zero, -8
; LA64-NEXT:    lu32i.d $a1, 0
; LA64-NEXT:    and $a0, $a0, $a1
; LA64-NEXT:    srli.d $a0, $a0, 3
; LA64-NEXT:    jirl $zero, $ra, 0
  %lshr = lshr i32 %x, 3
  ret i32 %lshr
}

define i64 @lshr_i64_3(i64 %x) {
; LA32-LABEL: lshr_i64_3:
; LA32:       # %bb.0:
; LA32-NEXT:    srli.w $a0, $a0, 3
; LA32-NEXT:    slli.w $a2, $a1, 29
; LA32-NEXT:    or $a0, $a0, $a2
; LA32-NEXT:    srli.w $a1, $a1, 3
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: lshr_i64_3:
; LA64:       # %bb.0:
; LA64-NEXT:    srli.d $a0, $a0, 3
; LA64-NEXT:    jirl $zero, $ra, 0
  %lshr = lshr i64 %x, 3
  ret i64 %lshr
}
