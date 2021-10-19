; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -verify-machineinstrs < %s | FileCheck %s -check-prefix=NO_SCALAR_INC
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -mattr=+use-scalar-inc-vl -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 -verify-machineinstrs < %s | FileCheck %s

define <vscale x 8 x i16> @inch_vec(<vscale x 8 x i16> %a) {
; NO_SCALAR_INC-LABEL: inch_vec:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    inch z0.h
; NO_SCALAR_INC-NEXT:    ret
;
; CHECK-LABEL: inch_vec:
; CHECK:       // %bb.0:
; CHECK-NEXT:    inch z0.h
; CHECK-NEXT:    ret
  %vscale = call i16 @llvm.vscale.i16()
  %mul = mul i16 %vscale, 8
  %vl = insertelement <vscale x 8 x i16> undef, i16 %mul, i32 0
  %vl.splat = shufflevector <vscale x 8 x i16> %vl, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res = add <vscale x 8 x i16> %a, %vl.splat
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @incw_vec(<vscale x 4 x i32> %a) {
; NO_SCALAR_INC-LABEL: incw_vec:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    incw z0.s
; NO_SCALAR_INC-NEXT:    ret
;
; CHECK-LABEL: incw_vec:
; CHECK:       // %bb.0:
; CHECK-NEXT:    incw z0.s
; CHECK-NEXT:    ret
  %vscale = call i32 @llvm.vscale.i32()
  %mul = mul i32 %vscale, 4
  %vl = insertelement <vscale x 4 x i32> undef, i32 %mul, i32 0
  %vl.splat = shufflevector <vscale x 4 x i32> %vl, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res = add <vscale x 4 x i32> %a, %vl.splat
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @incd_vec(<vscale x 2 x i64> %a) {
; NO_SCALAR_INC-LABEL: incd_vec:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    incd z0.d
; NO_SCALAR_INC-NEXT:    ret
;
; CHECK-LABEL: incd_vec:
; CHECK:       // %bb.0:
; CHECK-NEXT:    incd z0.d
; CHECK-NEXT:    ret
  %vscale = call i64 @llvm.vscale.i64()
  %mul = mul i64 %vscale, 2
  %vl = insertelement <vscale x 2 x i64> undef, i64 %mul, i32 0
  %vl.splat = shufflevector <vscale x 2 x i64> %vl, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res = add <vscale x 2 x i64> %a, %vl.splat
  ret <vscale x 2 x i64> %res
}

define <vscale x 8 x i16> @dech_vec(<vscale x 8 x i16> %a) {
; NO_SCALAR_INC-LABEL: dech_vec:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    dech z0.h, all, mul #2
; NO_SCALAR_INC-NEXT:    ret
;
; CHECK-LABEL: dech_vec:
; CHECK:       // %bb.0:
; CHECK-NEXT:    dech z0.h, all, mul #2
; CHECK-NEXT:    ret
  %vscale = call i16 @llvm.vscale.i16()
  %mul = mul i16 %vscale, 16
  %vl = insertelement <vscale x 8 x i16> undef, i16 %mul, i32 0
  %vl.splat = shufflevector <vscale x 8 x i16> %vl, <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %res = sub <vscale x 8 x i16> %a, %vl.splat
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @decw_vec(<vscale x 4 x i32> %a) {
; NO_SCALAR_INC-LABEL: decw_vec:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    decw z0.s, all, mul #4
; NO_SCALAR_INC-NEXT:    ret
;
; CHECK-LABEL: decw_vec:
; CHECK:       // %bb.0:
; CHECK-NEXT:    decw z0.s, all, mul #4
; CHECK-NEXT:    ret
  %vscale = call i32 @llvm.vscale.i32()
  %mul = mul i32 %vscale, 16
  %vl = insertelement <vscale x 4 x i32> undef, i32 %mul, i32 0
  %vl.splat = shufflevector <vscale x 4 x i32> %vl, <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %res = sub <vscale x 4 x i32> %a, %vl.splat
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @decd_vec(<vscale x 2 x i64> %a) {
; NO_SCALAR_INC-LABEL: decd_vec:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    decd z0.d, all, mul #8
; NO_SCALAR_INC-NEXT:    ret
;
; CHECK-LABEL: decd_vec:
; CHECK:       // %bb.0:
; CHECK-NEXT:    decd z0.d, all, mul #8
; CHECK-NEXT:    ret
  %vscale = call i64 @llvm.vscale.i64()
  %mul = mul i64 %vscale, 16
  %vl = insertelement <vscale x 2 x i64> undef, i64 %mul, i32 0
  %vl.splat = shufflevector <vscale x 2 x i64> %vl, <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %res = sub <vscale x 2 x i64> %a, %vl.splat
  ret <vscale x 2 x i64> %res
}

; NOTE: As there is no need for the predicate pattern we
; fall back to using ADDVL with its larger immediate range.
define i64 @incb_scalar_i64(i64 %a) {
; NO_SCALAR_INC-LABEL: incb_scalar_i64:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    addvl x0, x0, #1
; NO_SCALAR_INC-NEXT:    ret
;
; CHECK-LABEL: incb_scalar_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    addvl x0, x0, #1
; CHECK-NEXT:    ret
  %vscale = call i64 @llvm.vscale.i64()
  %mul = mul i64 %vscale, 16
  %add = add i64 %a, %mul
  ret i64 %add
}

define i64 @inch_scalar_i64(i64 %a) {
; NO_SCALAR_INC-LABEL: inch_scalar_i64:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    cnth x8
; NO_SCALAR_INC-NEXT:    add x0, x0, x8
; NO_SCALAR_INC-NEXT:    ret
;
; CHECK-LABEL: inch_scalar_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    inch x0
; CHECK-NEXT:    ret
  %vscale = call i64 @llvm.vscale.i64()
  %mul = mul i64 %vscale, 8
  %add = add i64 %a, %mul
  ret i64 %add
}

define i64 @incw_scalar_i64(i64 %a) {
; NO_SCALAR_INC-LABEL: incw_scalar_i64:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    cntw x8
; NO_SCALAR_INC-NEXT:    add x0, x0, x8
; NO_SCALAR_INC-NEXT:    ret
;
; CHECK-LABEL: incw_scalar_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    incw x0
; CHECK-NEXT:    ret
  %vscale = call i64 @llvm.vscale.i64()
  %mul = mul i64 %vscale, 4
  %add = add i64 %a, %mul
  ret i64 %add
}

define i64 @incd_scalar_i64(i64 %a) {
; NO_SCALAR_INC-LABEL: incd_scalar_i64:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    cntd x8
; NO_SCALAR_INC-NEXT:    add x0, x0, x8
; NO_SCALAR_INC-NEXT:    ret
;
; CHECK-LABEL: incd_scalar_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    incd x0
; CHECK-NEXT:    ret
  %vscale = call i64 @llvm.vscale.i64()
  %mul = mul i64 %vscale, 2
  %add = add i64 %a, %mul
  ret i64 %add
}

; NOTE: As there is no need for the predicate pattern we
; fall back to using ADDVL with its larger immediate range.
define i64 @decb_scalar_i64(i64 %a) {
; NO_SCALAR_INC-LABEL: decb_scalar_i64:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    addvl x0, x0, #-2
; NO_SCALAR_INC-NEXT:    ret
;
; CHECK-LABEL: decb_scalar_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    addvl x0, x0, #-2
; CHECK-NEXT:    ret
  %vscale = call i64 @llvm.vscale.i64()
  %mul = mul i64 %vscale, 32
  %sub = sub i64 %a, %mul
  ret i64 %sub
}

define i64 @dech_scalar_i64(i64 %a) {
; NO_SCALAR_INC-LABEL: dech_scalar_i64:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    cnth x8, all, mul #3
; NO_SCALAR_INC-NEXT:    neg x8, x8
; NO_SCALAR_INC-NEXT:    add x0, x0, x8
; NO_SCALAR_INC-NEXT:    ret
;
; CHECK-LABEL: dech_scalar_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    dech x0, all, mul #3
; CHECK-NEXT:    ret
  %vscale = call i64 @llvm.vscale.i64()
  %mul = mul i64 %vscale, 24
  %sub = sub i64 %a, %mul
  ret i64 %sub
}

define i64 @decw_scalar_i64(i64 %a) {
; NO_SCALAR_INC-LABEL: decw_scalar_i64:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    cntw x8, all, mul #3
; NO_SCALAR_INC-NEXT:    neg x8, x8
; NO_SCALAR_INC-NEXT:    add x0, x0, x8
; NO_SCALAR_INC-NEXT:    ret
;
; CHECK-LABEL: decw_scalar_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    decw x0, all, mul #3
; CHECK-NEXT:    ret
  %vscale = call i64 @llvm.vscale.i64()
  %mul = mul i64 %vscale, 12
  %sub = sub i64 %a, %mul
  ret i64 %sub
}

define i64 @decd_scalar_i64(i64 %a) {
; NO_SCALAR_INC-LABEL: decd_scalar_i64:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    cntd x8, all, mul #3
; NO_SCALAR_INC-NEXT:    neg x8, x8
; NO_SCALAR_INC-NEXT:    add x0, x0, x8
; NO_SCALAR_INC-NEXT:    ret
;
; CHECK-LABEL: decd_scalar_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    decd x0, all, mul #3
; CHECK-NEXT:    ret
  %vscale = call i64 @llvm.vscale.i64()
  %mul = mul i64 %vscale, 6
  %sub = sub i64 %a, %mul
  ret i64 %sub
}

; NOTE: As there is no need for the predicate pattern we
; fall back to using ADDVL with its larger immediate range.
define i32 @incb_scalar_i32(i32 %a) {
; NO_SCALAR_INC-LABEL: incb_scalar_i32:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    // kill: def $w0 killed $w0 def $x0
; NO_SCALAR_INC-NEXT:    addvl x0, x0, #3
; NO_SCALAR_INC-NEXT:    // kill: def $w0 killed $w0 killed $x0
; NO_SCALAR_INC-NEXT:    ret

; CHECK-LABEL: incb_scalar_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $w0 killed $w0 def $x0
; CHECK-NEXT:    addvl x0, x0, #3
; CHECK-NEXT:    // kill: def $w0 killed $w0 killed $x0
; CHECK-NEXT:    ret
  %vscale = call i64 @llvm.vscale.i64()
  %mul = mul i64 %vscale, 48
  %vl = trunc i64 %mul to i32
  %add = add i32 %a, %vl
  ret i32 %add
}

define i32 @inch_scalar_i32(i32 %a) {
; NO_SCALAR_INC-LABEL: inch_scalar_i32:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    cnth x8, all, mul #7
; NO_SCALAR_INC-NEXT:    add w0, w0, w8
; NO_SCALAR_INC-NEXT:    ret

; CHECK-LABEL: inch_scalar_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $w0 killed $w0 def $x0
; CHECK-NEXT:    inch x0, all, mul #7
; CHECK-NEXT:    // kill: def $w0 killed $w0 killed $x0
; CHECK-NEXT:    ret
  %vscale = call i64 @llvm.vscale.i64()
  %mul = mul i64 %vscale, 56
  %vl = trunc i64 %mul to i32
  %add = add i32 %a, %vl
  ret i32 %add
}

define i32 @incw_scalar_i32(i32 %a) {
; NO_SCALAR_INC-LABEL: incw_scalar_i32:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    cntw x8, all, mul #7
; NO_SCALAR_INC-NEXT:    add w0, w0, w8
; NO_SCALAR_INC-NEXT:    ret

; CHECK-LABEL: incw_scalar_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $w0 killed $w0 def $x0
; CHECK-NEXT:    incw x0, all, mul #7
; CHECK-NEXT:    // kill: def $w0 killed $w0 killed $x0
; CHECK-NEXT:    ret
  %vscale = call i64 @llvm.vscale.i64()
  %mul = mul i64 %vscale, 28
  %vl = trunc i64 %mul to i32
  %add = add i32 %a, %vl
  ret i32 %add
}

define i32 @incd_scalar_i32(i32 %a) {
; NO_SCALAR_INC-LABEL: incd_scalar_i32:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    cntd x8, all, mul #7
; NO_SCALAR_INC-NEXT:    add w0, w0, w8
; NO_SCALAR_INC-NEXT:    ret

; CHECK-LABEL: incd_scalar_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $w0 killed $w0 def $x0
; CHECK-NEXT:    incd x0, all, mul #7
; CHECK-NEXT:    // kill: def $w0 killed $w0 killed $x0
; CHECK-NEXT:    ret
  %vscale = call i64 @llvm.vscale.i64()
  %mul = mul i64 %vscale, 14
  %vl = trunc i64 %mul to i32
  %add = add i32 %a, %vl
  ret i32 %add
}

; NOTE: As there is no need for the predicate pattern we
; fall back to using ADDVL with its larger immediate range.
define i32 @decb_scalar_i32(i32 %a) {
; NO_SCALAR_INC-LABEL: decb_scalar_i32:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    // kill: def $w0 killed $w0 def $x0
; NO_SCALAR_INC-NEXT:    addvl x0, x0, #-4
; NO_SCALAR_INC-NEXT:    // kill: def $w0 killed $w0 killed $x0
; NO_SCALAR_INC-NEXT:    ret

; CHECK-LABEL: decb_scalar_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $w0 killed $w0 def $x0
; CHECK-NEXT:    addvl x0, x0, #-4
; CHECK-NEXT:    // kill: def $w0 killed $w0 killed $x0
; CHECK-NEXT:    ret
  %vscale = call i64 @llvm.vscale.i64()
  %mul = mul i64 %vscale, 64
  %vl = trunc i64 %mul to i32
  %sub = sub i32 %a, %vl
  ret i32 %sub
}

define i32 @dech_scalar_i32(i32 %a) {
; NO_SCALAR_INC-LABEL: dech_scalar_i32:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    cnth x8
; NO_SCALAR_INC-NEXT:    neg x8, x8
; NO_SCALAR_INC-NEXT:    add w0, w0, w8
; NO_SCALAR_INC-NEXT:    ret

; CHECK-LABEL: dech_scalar_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $w0 killed $w0 def $x0
; CHECK-NEXT:    dech x0
; CHECK-NEXT:    // kill: def $w0 killed $w0 killed $x0
; CHECK-NEXT:    ret
  %vscale = call i64 @llvm.vscale.i64()
  %mul = mul i64 %vscale, 8
  %vl = trunc i64 %mul to i32
  %sub = sub i32 %a, %vl
  ret i32 %sub
}

define i32 @decw_scalar_i32(i32 %a) {
; NO_SCALAR_INC-LABEL: decw_scalar_i32:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    cntw x8
; NO_SCALAR_INC-NEXT:    neg x8, x8
; NO_SCALAR_INC-NEXT:    add w0, w0, w8
; NO_SCALAR_INC-NEXT:    ret

; CHECK-LABEL: decw_scalar_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $w0 killed $w0 def $x0
; CHECK-NEXT:    decw x0
; CHECK-NEXT:    // kill: def $w0 killed $w0 killed $x0
; CHECK-NEXT:    ret
  %vscale = call i64 @llvm.vscale.i64()
  %mul = mul i64 %vscale, 4
  %vl = trunc i64 %mul to i32
  %sub = sub i32 %a, %vl
  ret i32 %sub
}

define i32 @decd_scalar_i32(i32 %a) {
; NO_SCALAR_INC-LABEL: decd_scalar_i32:
; NO_SCALAR_INC:       // %bb.0:
; NO_SCALAR_INC-NEXT:    cntd x8
; NO_SCALAR_INC-NEXT:    neg x8, x8
; NO_SCALAR_INC-NEXT:    add w0, w0, w8
; NO_SCALAR_INC-NEXT:    ret
;
; CHECK-LABEL: decd_scalar_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $w0 killed $w0 def $x0
; CHECK-NEXT:    decd x0
; CHECK-NEXT:    // kill: def $w0 killed $w0 killed $x0
; CHECK-NEXT:    ret
  %vscale = call i64 @llvm.vscale.i64()
  %mul = mul i64 %vscale, 2
  %vl = trunc i64 %mul to i32
  %sub = sub i32 %a, %vl
  ret i32 %sub
}

declare i16 @llvm.vscale.i16()
declare i32 @llvm.vscale.i32()
declare i64 @llvm.vscale.i64()
