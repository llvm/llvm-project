; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

;; These tests should run for all targets

;;===-- Basic instruction selection tests ---------------------------------===;;


;;; i64

define i64 @add_i64(i64 %a, i64 %b) {
; CHECK: add.s64 %rd{{[0-9]+}}, %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: ret
  %ret = add i64 %a, %b
  ret i64 %ret
}

define i64 @sub_i64(i64 %a, i64 %b) {
; CHECK: sub.s64 %rd{{[0-9]+}}, %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: ret
  %ret = sub i64 %a, %b
  ret i64 %ret
}

define i64 @mul_i64(i64 %a, i64 %b) {
; CHECK: mul.lo.s64 %rd{{[0-9]+}}, %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: ret
  %ret = mul i64 %a, %b
  ret i64 %ret
}

define i64 @umul_lohi_i64(i64 %a) {
; CHECK-LABEL: umul_lohi_i64(
entry:
  %0 = zext i64 %a to i128
  %1 = mul i128 %0, 288
; CHECK: mul.lo.{{u|s}}64
; CHECK: mul.hi.{{u|s}}64
  %2 = lshr i128 %1, 1
  %3 = trunc i128 %2 to i64
  ret i64 %3
}

define i64 @smul_lohi_i64(i64 %a) {
; CHECK-LABEL: smul_lohi_i64(
entry:
  %0 = sext i64 %a to i128
  %1 = mul i128 %0, 288
; CHECK: mul.lo.{{u|s}}64
; CHECK: mul.hi.{{u|s}}64
  %2 = ashr i128 %1, 1
  %3 = trunc i128 %2 to i64
  ret i64 %3
}

define i64 @sdiv_i64(i64 %a, i64 %b) {
; CHECK: div.s64 %rd{{[0-9]+}}, %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: ret
  %ret = sdiv i64 %a, %b
  ret i64 %ret
}

define i64 @udiv_i64(i64 %a, i64 %b) {
; CHECK: div.u64 %rd{{[0-9]+}}, %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: ret
  %ret = udiv i64 %a, %b
  ret i64 %ret
}

define i64 @srem_i64(i64 %a, i64 %b) {
; CHECK: rem.s64 %rd{{[0-9]+}}, %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: ret
  %ret = srem i64 %a, %b
  ret i64 %ret
}

define i64 @urem_i64(i64 %a, i64 %b) {
; CHECK: rem.u64 %rd{{[0-9]+}}, %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: ret
  %ret = urem i64 %a, %b
  ret i64 %ret
}

define i64 @and_i64(i64 %a, i64 %b) {
; CHECK: and.b64 %rd{{[0-9]+}}, %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: ret
  %ret = and i64 %a, %b
  ret i64 %ret
}

define i64 @or_i64(i64 %a, i64 %b) {
; CHECK: or.b64 %rd{{[0-9]+}}, %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: ret
  %ret = or i64 %a, %b
  ret i64 %ret
}

define i64 @xor_i64(i64 %a, i64 %b) {
; CHECK: xor.b64 %rd{{[0-9]+}}, %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: ret
  %ret = xor i64 %a, %b
  ret i64 %ret
}

define i64 @shl_i64(i64 %a, i64 %b) {
; PTX requires 32-bit shift amount
; CHECK: shl.b64 %rd{{[0-9]+}}, %rd{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: ret
  %ret = shl i64 %a, %b
  ret i64 %ret
}

define i64 @ashr_i64(i64 %a, i64 %b) {
; PTX requires 32-bit shift amount
; CHECK: shr.s64 %rd{{[0-9]+}}, %rd{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: ret
  %ret = ashr i64 %a, %b
  ret i64 %ret
}

define i64 @lshr_i64(i64 %a, i64 %b) {
; PTX requires 32-bit shift amount
; CHECK: shr.u64 %rd{{[0-9]+}}, %rd{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: ret
  %ret = lshr i64 %a, %b
  ret i64 %ret
}


;;; i32

define i32 @add_i32(i32 %a, i32 %b) {
; CHECK: add.s32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: ret
  %ret = add i32 %a, %b
  ret i32 %ret
}

define i32 @sub_i32(i32 %a, i32 %b) {
; CHECK: sub.s32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: ret
  %ret = sub i32 %a, %b
  ret i32 %ret
}

define i32 @mul_i32(i32 %a, i32 %b) {
; CHECK: mul.lo.s32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: ret
  %ret = mul i32 %a, %b
  ret i32 %ret
}

define i32 @sdiv_i32(i32 %a, i32 %b) {
; CHECK: div.s32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: ret
  %ret = sdiv i32 %a, %b
  ret i32 %ret
}

define i32 @udiv_i32(i32 %a, i32 %b) {
; CHECK: div.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: ret
  %ret = udiv i32 %a, %b
  ret i32 %ret
}

define i32 @srem_i32(i32 %a, i32 %b) {
; CHECK: rem.s32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: ret
  %ret = srem i32 %a, %b
  ret i32 %ret
}

define i32 @urem_i32(i32 %a, i32 %b) {
; CHECK: rem.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: ret
  %ret = urem i32 %a, %b
  ret i32 %ret
}

define i32 @and_i32(i32 %a, i32 %b) {
; CHECK: and.b32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: ret
  %ret = and i32 %a, %b
  ret i32 %ret
}

define i32 @or_i32(i32 %a, i32 %b) {
; CHECK: or.b32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: ret
  %ret = or i32 %a, %b
  ret i32 %ret
}

define i32 @xor_i32(i32 %a, i32 %b) {
; CHECK: xor.b32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: ret
  %ret = xor i32 %a, %b
  ret i32 %ret
}

define i32 @shl_i32(i32 %a, i32 %b) {
; CHECK: shl.b32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: ret
  %ret = shl i32 %a, %b
  ret i32 %ret
}

define i32 @ashr_i32(i32 %a, i32 %b) {
; CHECK: shr.s32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: ret
  %ret = ashr i32 %a, %b
  ret i32 %ret
}

define i32 @lshr_i32(i32 %a, i32 %b) {
; CHECK: shr.u32 %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: ret
  %ret = lshr i32 %a, %b
  ret i32 %ret
}

;;; i16

define i16 @add_i16(i16 %a, i16 %b) {
; CHECK: add.s16 %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: ret
  %ret = add i16 %a, %b
  ret i16 %ret
}

define i16 @sub_i16(i16 %a, i16 %b) {
; CHECK: sub.s16 %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: ret
  %ret = sub i16 %a, %b
  ret i16 %ret
}

define i16 @mul_i16(i16 %a, i16 %b) {
; CHECK: mul.lo.s16 %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: ret
  %ret = mul i16 %a, %b
  ret i16 %ret
}

define i16 @sdiv_i16(i16 %a, i16 %b) {
; CHECK: div.s16 %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: ret
  %ret = sdiv i16 %a, %b
  ret i16 %ret
}

define i16 @udiv_i16(i16 %a, i16 %b) {
; CHECK: div.u16 %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: ret
  %ret = udiv i16 %a, %b
  ret i16 %ret
}

define i16 @srem_i16(i16 %a, i16 %b) {
; CHECK: rem.s16 %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: ret
  %ret = srem i16 %a, %b
  ret i16 %ret
}

define i16 @urem_i16(i16 %a, i16 %b) {
; CHECK: rem.u16 %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: ret
  %ret = urem i16 %a, %b
  ret i16 %ret
}

define i16 @and_i16(i16 %a, i16 %b) {
; CHECK: and.b16 %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: ret
  %ret = and i16 %a, %b
  ret i16 %ret
}

define i16 @or_i16(i16 %a, i16 %b) {
; CHECK: or.b16 %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: ret
  %ret = or i16 %a, %b
  ret i16 %ret
}

define i16 @xor_i16(i16 %a, i16 %b) {
; CHECK: xor.b16 %rs{{[0-9]+}}, %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: ret
  %ret = xor i16 %a, %b
  ret i16 %ret
}

define i16 @shl_i16(i16 %a, i16 %b) {
; PTX requires 32-bit shift amount
; CHECK: shl.b16 %rs{{[0-9]+}}, %rs{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: ret
  %ret = shl i16 %a, %b
  ret i16 %ret
}

define i16 @ashr_i16(i16 %a, i16 %b) {
; PTX requires 32-bit shift amount
; CHECK: shr.s16 %rs{{[0-9]+}}, %rs{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: ret
  %ret = ashr i16 %a, %b
  ret i16 %ret
}

define i16 @lshr_i16(i16 %a, i16 %b) {
; PTX requires 32-bit shift amount
; CHECK: shr.u16 %rs{{[0-9]+}}, %rs{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: ret
  %ret = lshr i16 %a, %b
  ret i16 %ret
}
