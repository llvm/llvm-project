; RUN: llc --mtriple=loongarch32 < %s | FileCheck %s --check-prefix=LA32
; RUN: llc --mtriple=loongarch64 < %s | FileCheck %s --check-prefix=LA64

;; Test sext/zext/trunc

define i8 @sext_i1_to_i8(i1 %a) {
; LA32-LABEL: sext_i1_to_i8:
; LA32:       # %bb.0:
; LA32-NEXT:    andi $a0, $a0, 1
; LA32-NEXT:    sub.w $a0, $zero, $a0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: sext_i1_to_i8:
; LA64:       # %bb.0:
; LA64-NEXT:    andi $a0, $a0, 1
; LA64-NEXT:    sub.d $a0, $zero, $a0
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = sext i1 %a to i8
  ret i8 %1
}

define i16 @sext_i1_to_i16(i1 %a) {
; LA32-LABEL: sext_i1_to_i16:
; LA32:       # %bb.0:
; LA32-NEXT:    andi $a0, $a0, 1
; LA32-NEXT:    sub.w $a0, $zero, $a0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: sext_i1_to_i16:
; LA64:       # %bb.0:
; LA64-NEXT:    andi $a0, $a0, 1
; LA64-NEXT:    sub.d $a0, $zero, $a0
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = sext i1 %a to i16
  ret i16 %1
}

define i32 @sext_i1_to_i32(i1 %a) {
; LA32-LABEL: sext_i1_to_i32:
; LA32:       # %bb.0:
; LA32-NEXT:    andi $a0, $a0, 1
; LA32-NEXT:    sub.w $a0, $zero, $a0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: sext_i1_to_i32:
; LA64:       # %bb.0:
; LA64-NEXT:    andi $a0, $a0, 1
; LA64-NEXT:    sub.d $a0, $zero, $a0
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = sext i1 %a to i32
  ret i32 %1
}

define i64 @sext_i1_to_i64(i1 %a) {
; LA32-LABEL: sext_i1_to_i64:
; LA32:       # %bb.0:
; LA32-NEXT:    andi $a0, $a0, 1
; LA32-NEXT:    sub.w $a0, $zero, $a0
; LA32-NEXT:    move $a1, $a0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: sext_i1_to_i64:
; LA64:       # %bb.0:
; LA64-NEXT:    andi $a0, $a0, 1
; LA64-NEXT:    sub.d $a0, $zero, $a0
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = sext i1 %a to i64
  ret i64 %1
}

define i16 @sext_i8_to_i16(i8 %a) {
; LA32-LABEL: sext_i8_to_i16:
; LA32:       # %bb.0:
; LA32-NEXT:    ext.w.b $a0, $a0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: sext_i8_to_i16:
; LA64:       # %bb.0:
; LA64-NEXT:    ext.w.b $a0, $a0
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = sext i8 %a to i16
  ret i16 %1
}

define i32 @sext_i8_to_i32(i8 %a) {
; LA32-LABEL: sext_i8_to_i32:
; LA32:       # %bb.0:
; LA32-NEXT:    ext.w.b $a0, $a0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: sext_i8_to_i32:
; LA64:       # %bb.0:
; LA64-NEXT:    ext.w.b $a0, $a0
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = sext i8 %a to i32
  ret i32 %1
}

define i64 @sext_i8_to_i64(i8 %a) {
; LA32-LABEL: sext_i8_to_i64:
; LA32:       # %bb.0:
; LA32-NEXT:    ext.w.b $a0, $a0
; LA32-NEXT:    srai.w $a1, $a0, 31
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: sext_i8_to_i64:
; LA64:       # %bb.0:
; LA64-NEXT:    ext.w.b $a0, $a0
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = sext i8 %a to i64
  ret i64 %1
}

define i32 @sext_i16_to_i32(i16 %a) {
; LA32-LABEL: sext_i16_to_i32:
; LA32:       # %bb.0:
; LA32-NEXT:    ext.w.h $a0, $a0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: sext_i16_to_i32:
; LA64:       # %bb.0:
; LA64-NEXT:    ext.w.h $a0, $a0
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = sext i16 %a to i32
  ret i32 %1
}

define i64 @sext_i16_to_i64(i16 %a) {
; LA32-LABEL: sext_i16_to_i64:
; LA32:       # %bb.0:
; LA32-NEXT:    ext.w.h $a0, $a0
; LA32-NEXT:    srai.w $a1, $a0, 31
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: sext_i16_to_i64:
; LA64:       # %bb.0:
; LA64-NEXT:    ext.w.h $a0, $a0
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = sext i16 %a to i64
  ret i64 %1
}

define i64 @sext_i32_to_i64(i32 %a) {
; LA32-LABEL: sext_i32_to_i64:
; LA32:       # %bb.0:
; LA32-NEXT:    srai.w $a1, $a0, 31
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: sext_i32_to_i64:
; LA64:       # %bb.0:
; LA64-NEXT:    addi.w $a0, $a0, 0
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = sext i32 %a to i64
  ret i64 %1
}

define i8 @zext_i1_to_i8(i1 %a) {
; LA32-LABEL: zext_i1_to_i8:
; LA32:       # %bb.0:
; LA32-NEXT:    andi $a0, $a0, 1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: zext_i1_to_i8:
; LA64:       # %bb.0:
; LA64-NEXT:    andi $a0, $a0, 1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = zext i1 %a to i8
  ret i8 %1
}

define i16 @zext_i1_to_i16(i1 %a) {
; LA32-LABEL: zext_i1_to_i16:
; LA32:       # %bb.0:
; LA32-NEXT:    andi $a0, $a0, 1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: zext_i1_to_i16:
; LA64:       # %bb.0:
; LA64-NEXT:    andi $a0, $a0, 1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = zext i1 %a to i16
  ret i16 %1
}

define i32 @zext_i1_to_i32(i1 %a) {
; LA32-LABEL: zext_i1_to_i32:
; LA32:       # %bb.0:
; LA32-NEXT:    andi $a0, $a0, 1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: zext_i1_to_i32:
; LA64:       # %bb.0:
; LA64-NEXT:    andi $a0, $a0, 1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = zext i1 %a to i32
  ret i32 %1
}

define i64 @zext_i1_to_i64(i1 %a) {
; LA32-LABEL: zext_i1_to_i64:
; LA32:       # %bb.0:
; LA32-NEXT:    andi $a0, $a0, 1
; LA32-NEXT:    move $a1, $zero
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: zext_i1_to_i64:
; LA64:       # %bb.0:
; LA64-NEXT:    andi $a0, $a0, 1
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = zext i1 %a to i64
  ret i64 %1
}

define i16 @zext_i8_to_i16(i8 %a) {
; LA32-LABEL: zext_i8_to_i16:
; LA32:       # %bb.0:
; LA32-NEXT:    andi $a0, $a0, 255
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: zext_i8_to_i16:
; LA64:       # %bb.0:
; LA64-NEXT:    andi $a0, $a0, 255
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = zext i8 %a to i16
  ret i16 %1
}

define i32 @zext_i8_to_i32(i8 %a) {
; LA32-LABEL: zext_i8_to_i32:
; LA32:       # %bb.0:
; LA32-NEXT:    andi $a0, $a0, 255
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: zext_i8_to_i32:
; LA64:       # %bb.0:
; LA64-NEXT:    andi $a0, $a0, 255
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = zext i8 %a to i32
  ret i32 %1
}

define i64 @zext_i8_to_i64(i8 %a) {
; LA32-LABEL: zext_i8_to_i64:
; LA32:       # %bb.0:
; LA32-NEXT:    andi $a0, $a0, 255
; LA32-NEXT:    move $a1, $zero
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: zext_i8_to_i64:
; LA64:       # %bb.0:
; LA64-NEXT:    andi $a0, $a0, 255
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = zext i8 %a to i64
  ret i64 %1
}

define i32 @zext_i16_to_i32(i16 %a) {
; LA32-LABEL: zext_i16_to_i32:
; LA32:       # %bb.0:
; LA32-NEXT:    bstrpick.w $a0, $a0, 15, 0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: zext_i16_to_i32:
; LA64:       # %bb.0:
; LA64-NEXT:    bstrpick.d $a0, $a0, 15, 0
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = zext i16 %a to i32
  ret i32 %1
}

define i64 @zext_i16_to_i64(i16 %a) {
; LA32-LABEL: zext_i16_to_i64:
; LA32:       # %bb.0:
; LA32-NEXT:    bstrpick.w $a0, $a0, 15, 0
; LA32-NEXT:    move $a1, $zero
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: zext_i16_to_i64:
; LA64:       # %bb.0:
; LA64-NEXT:    bstrpick.d $a0, $a0, 15, 0
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = zext i16 %a to i64
  ret i64 %1
}

define i64 @zext_i32_to_i64(i32 %a) {
; LA32-LABEL: zext_i32_to_i64:
; LA32:       # %bb.0:
; LA32-NEXT:    move $a1, $zero
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: zext_i32_to_i64:
; LA64:       # %bb.0:
; LA64-NEXT:    bstrpick.d $a0, $a0, 31, 0
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = zext i32 %a to i64
  ret i64 %1
}

define i1 @trunc_i8_to_i1(i8 %a) {
; LA32-LABEL: trunc_i8_to_i1:
; LA32:       # %bb.0:
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: trunc_i8_to_i1:
; LA64:       # %bb.0:
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = trunc i8 %a to i1
  ret i1 %1
}

define i1 @trunc_i16_to_i1(i16 %a) {
; LA32-LABEL: trunc_i16_to_i1:
; LA32:       # %bb.0:
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: trunc_i16_to_i1:
; LA64:       # %bb.0:
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = trunc i16 %a to i1
  ret i1 %1
}

define i1 @trunc_i32_to_i1(i32 %a) {
; LA32-LABEL: trunc_i32_to_i1:
; LA32:       # %bb.0:
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: trunc_i32_to_i1:
; LA64:       # %bb.0:
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = trunc i32 %a to i1
  ret i1 %1
}

define i1 @trunc_i64_to_i1(i64 %a) {
; LA32-LABEL: trunc_i64_to_i1:
; LA32:       # %bb.0:
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: trunc_i64_to_i1:
; LA64:       # %bb.0:
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = trunc i64 %a to i1
  ret i1 %1
}

define i8 @trunc_i16_to_i8(i16 %a) {
; LA32-LABEL: trunc_i16_to_i8:
; LA32:       # %bb.0:
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: trunc_i16_to_i8:
; LA64:       # %bb.0:
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = trunc i16 %a to i8
  ret i8 %1
}

define i8 @trunc_i32_to_i8(i32 %a) {
; LA32-LABEL: trunc_i32_to_i8:
; LA32:       # %bb.0:
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: trunc_i32_to_i8:
; LA64:       # %bb.0:
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = trunc i32 %a to i8
  ret i8 %1
}

define i8 @trunc_i64_to_i8(i64 %a) {
; LA32-LABEL: trunc_i64_to_i8:
; LA32:       # %bb.0:
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: trunc_i64_to_i8:
; LA64:       # %bb.0:
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = trunc i64 %a to i8
  ret i8 %1
}

define i16 @trunc_i32_to_i16(i32 %a) {
; LA32-LABEL: trunc_i32_to_i16:
; LA32:       # %bb.0:
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: trunc_i32_to_i16:
; LA64:       # %bb.0:
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = trunc i32 %a to i16
  ret i16 %1
}

define i16 @trunc_i64_to_i16(i64 %a) {
; LA32-LABEL: trunc_i64_to_i16:
; LA32:       # %bb.0:
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: trunc_i64_to_i16:
; LA64:       # %bb.0:
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = trunc i64 %a to i16
  ret i16 %1
}

define i32 @trunc_i64_to_i32(i64 %a) {
; LA32-LABEL: trunc_i64_to_i32:
; LA32:       # %bb.0:
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: trunc_i64_to_i32:
; LA64:       # %bb.0:
; LA64-NEXT:    jirl $zero, $ra, 0
  %1 = trunc i64 %a to i32
  ret i32 %1
}
