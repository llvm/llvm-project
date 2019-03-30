; RUN: llc -mtriple=thumbv8m.main -mcpu=cortex-m33 %s -arm-disable-cgp=false -o - | FileCheck %s

; CHECK: overflow_add
; CHECK: add
; CHECK: uxth
; CHECK: cmp
define zeroext i16 @overflow_add(i16 zeroext %a, i16 zeroext %b) {
  %add = add i16 %a, %b
  %or = or i16 %add, 1
  %cmp = icmp ugt i16 %or, 1024
  %res = select i1 %cmp, i16 2, i16 5
  ret i16 %res
}

; CHECK-LABEL: overflow_sub
; CHECK: sub
; CHECK: uxth
; CHECK: cmp
define zeroext i16 @overflow_sub(i16 zeroext %a, i16 zeroext %b) {
  %add = sub i16 %a, %b
  %or = or i16 %add, 1
  %cmp = icmp ugt i16 %or, 1024
  %res = select i1 %cmp, i16 2, i16 5
  ret i16 %res
}

; CHECK-LABEL: overflow_mul
; CHECK: mul
; CHECK: uxth
; CHECK: cmp
define zeroext i16 @overflow_mul(i16 zeroext %a, i16 zeroext %b) {
  %add = mul i16 %a, %b
  %or = or i16 %add, 1
  %cmp = icmp ugt i16 %or, 1024
  %res = select i1 %cmp, i16 2, i16 5
  ret i16 %res
}

; CHECK-LABEL: overflow_shl
; CHECK-COMMON: lsl
; CHECK-COMMON: uxth
; CHECK-COMMON: cmp
define zeroext i16 @overflow_shl(i16 zeroext %a, i16 zeroext %b) {
  %add = shl i16 %a, %b
  %or = or i16 %add, 1
  %cmp = icmp ugt i16 %or, 1024
  %res = select i1 %cmp, i16 2, i16 5
  ret i16 %res
}

; CHECK-LABEL: overflow_add_no_consts:
; CHECK:  add r0, r1
; CHECK:  uxtb [[EXT:r[0-9]+]], r0
; CHECK:  cmp [[EXT]], r2
; CHECK:  movhi r0, #8
define i32 @overflow_add_no_consts(i8 zeroext %a, i8 zeroext %b, i8 zeroext %limit) {
  %add = add i8 %a, %b
  %cmp = icmp ugt i8 %add, %limit
  %res = select i1 %cmp, i32 8, i32 16
  ret i32 %res
}

; CHECK-LABEL: overflow_add_const_limit:
; CHECK:  add r0, r1
; CHECK:  uxtb [[EXT:r[0-9]+]], r0
; CHECK:  cmp [[EXT]], #128
; CHECK:  movhi r0, #8
define i32 @overflow_add_const_limit(i8 zeroext %a, i8 zeroext %b) {
  %add = add i8 %a, %b
  %cmp = icmp ugt i8 %add, 128
  %res = select i1 %cmp, i32 8, i32 16
  ret i32 %res
}

; CHECK-LABEL: overflow_add_positive_const_limit:
; CHECK:  adds r0, #1
; CHECK:  uxtb [[EXT:r[0-9]+]], r0
; CHECK:  cmp [[EXT]], #128
; CHECK:  movhi r0, #8
define i32 @overflow_add_positive_const_limit(i8 zeroext %a) {
  %add = add i8 %a, 1
  %cmp = icmp ugt i8 %add, 128
  %res = select i1 %cmp, i32 8, i32 16
  ret i32 %res
}

; CHECK-LABEL: unsafe_add_underflow:
; CHECK: movs	r1, #16
; CHECK: cmp	r0, #1
; CHECK: it	eq
; CHECK: moveq	r1, #8
; CHECK: mov	r0, r1
define i32 @unsafe_add_underflow(i8 zeroext %a) {
  %add = add i8 %a, -2
  %cmp = icmp ugt i8 %add, 254
  %res = select i1 %cmp, i32 8, i32 16
  ret i32 %res
}

; CHECK-LABEL: safe_add_underflow:
; CHECK:      subs [[MINUS_1:r[0-9]+]], r0, #1
; CHECK-NOT:  uxtb
; CHECK:      cmp [[MINUS_1]], #254
; CHECK:      movhi r0, #8
define i32 @safe_add_underflow(i8 zeroext %a) {
  %add = add i8 %a, -1
  %cmp = icmp ugt i8 %add, 254
  %res = select i1 %cmp, i32 8, i32 16
  ret i32 %res
}

; CHECK-LABEL: safe_add_underflow_neg:
; CHECK:      subs [[MINUS_1:r[0-9]+]], r0, #2
; CHECK-NOT:  uxtb
; CHECK:      cmp [[MINUS_1]], #251
; CHECK:      movlo r0, #8
define i32 @safe_add_underflow_neg(i8 zeroext %a) {
  %add = add i8 %a, -2
  %cmp = icmp ule i8 %add, -6
  %res = select i1 %cmp, i32 8, i32 16
  ret i32 %res
}

; CHECK-LABEL: overflow_sub_negative_const_limit:
; CHECK:  adds r0, #1
; CHECK:  uxtb [[EXT:r[0-9]+]], r0
; CHECK:  cmp [[EXT]], #128
; CHECK:  movhi r0, #8
define i32 @overflow_sub_negative_const_limit(i8 zeroext %a) {
  %sub = sub i8 %a, -1
  %cmp = icmp ugt i8 %sub, 128
  %res = select i1 %cmp, i32 8, i32 16
  ret i32 %res
}

; CHECK-LABEL: unsafe_sub_underflow:
; CHECK:  subs r0, #6
; CHECK:  uxtb [[EXT:r[0-9]+]], r0
; CHECK:  cmp [[EXT]], #250
; CHECK:  movhi r0, #8
define i32 @unsafe_sub_underflow(i8 zeroext %a) {
  %sub = sub i8 %a, 6
  %cmp = icmp ugt i8 %sub, 250
  %res = select i1 %cmp, i32 8, i32 16
  ret i32 %res
}

; CHECK-LABEL: safe_sub_underflow:
; CHECK:      subs [[MINUS_1:r[0-9]+]], r0, #1
; CHECK-NOT:  uxtb
; CHECK:      cmp [[MINUS_1]], #255
; CHECK:      movlo r0, #8
define i32 @safe_sub_underflow(i8 zeroext %a) {
  %sub = sub i8 %a, 1
  %cmp = icmp ule i8 %sub, 254
  %res = select i1 %cmp, i32 8, i32 16
  ret i32 %res
}

; CHECK-LABEL: safe_sub_underflow_neg
; CHECK:      subs [[MINUS_1:r[0-9]+]], r0, #4
; CHECK-NOT:  uxtb
; CHECK:      cmp [[MINUS_1]], #250
; CHECK:      movhi r0, #8
define i32 @safe_sub_underflow_neg(i8 zeroext %a) {
  %sub = sub i8 %a, 4
  %cmp = icmp uge i8 %sub, -5
  %res = select i1 %cmp, i32 8, i32 16
  ret i32 %res
}

; CHECK-LABEL: unsafe_sub_underflow_neg
; CHECK:  subs r0, #4
; CHECK:  uxtb [[EXT:r[0-9]+]], r0
; CHECK:  cmp [[EXT]], #253
; CHECK:  movlo r0, #8
define i32 @unsafe_sub_underflow_neg(i8 zeroext %a) {
  %sub = sub i8 %a, 4
  %cmp = icmp ult i8 %sub, -3
  %res = select i1 %cmp, i32 8, i32 16
  ret i32 %res
}

; CHECK:      rsb.w [[RSUB:r[0-9]+]], r0, #248
; CHECK-NOT:  uxt
; CHECK:      cmp [[RSUB]], #252
define i32 @safe_sub_imm_var(i8* %b) {
entry:
  %0 = load i8, i8* %b, align 1
  %sub = sub nuw nsw i8 -8, %0
  %cmp = icmp ugt i8 %sub, 252
  %conv4 = zext i1 %cmp to i32
  ret i32 %conv4
}

; CHECK-LABEL: safe_sub_var_imm
; CHECK:      add.w [[ADD:r[0-9]+]], r0, #8
; CHECK-NOT:  uxt
; CHECK:      cmp [[ADD]], #252
define i32 @safe_sub_var_imm(i8* %b) {
entry:
  %0 = load i8, i8* %b, align 1
  %sub = sub nuw nsw i8 %0, -8
  %cmp = icmp ugt i8 %sub, 252
  %conv4 = zext i1 %cmp to i32
  ret i32 %conv4
}

; CHECK-LABEL: safe_add_imm_var
; CHECK:      add.w [[ADD:r[0-9]+]], r0, #129
; CHECK-NOT:  uxt
; CHECK:      cmp [[ADD]], #127
define i32 @safe_add_imm_var(i8* %b) {
entry:
  %0 = load i8, i8* %b, align 1
  %add = add nuw nsw i8 -127, %0
  %cmp = icmp ugt i8 %add, 127
  %conv4 = zext i1 %cmp to i32
  ret i32 %conv4
}

; CHECK-LABEL: safe_add_var_imm
; CHECK:      sub.w [[SUB:r[0-9]+]], r0, #127
; CHECK-NOT:  uxt
; CHECK:      cmp [[SUB]], #127
define i32 @safe_add_var_imm(i8* %b) {
entry:
  %0 = load i8, i8* %b, align 1
  %add = add nuw nsw i8 %0, -127
  %cmp = icmp ugt i8 %add, 127
  %conv4 = zext i1 %cmp to i32
  ret i32 %conv4
}

; CHECK-LABEL: convert_add_order
; CHECK: orr{{.*}}, #1
; CHECK: sub{{.*}}, #40
; CHECK-NOT: uxt
define i8 @convert_add_order(i8 zeroext %arg) {
  %mask.0 = and i8 %arg, 1
  %mask.1 = and i8 %arg, 2
  %shl = or i8 %arg, 1
  %add = add nuw i8 %shl, 10
  %cmp.0 = icmp ult i8 %add, 60
  %sub = add nsw i8 %shl, -40
  %cmp.1 = icmp ult i8 %sub, 20
  %mask.sel = select i1 %cmp.1, i8 %mask.0, i8 %mask.1
  %res = select i1 %cmp.0, i8 %mask.sel, i8 %arg
  ret i8 %res
}
