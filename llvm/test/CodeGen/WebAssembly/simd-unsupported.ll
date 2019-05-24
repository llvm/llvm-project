; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+unimplemented-simd128 | FileCheck %s

; Test that operations that are not supported by SIMD are properly
; unrolled.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; ==============================================================================
; 16 x i8
; ==============================================================================

; CHECK-LABEL: ctlz_v16i8:
; CHECK: i32.clz
declare <16 x i8> @llvm.ctlz.v16i8(<16 x i8>, i1)
define <16 x i8> @ctlz_v16i8(<16 x i8> %x) {
  %v = call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %x, i1 false)
  ret <16 x i8> %v
}

; CHECK-LABEL: ctlz_v16i8_undef:
; CHECK: i32.clz
define <16 x i8> @ctlz_v16i8_undef(<16 x i8> %x) {
  %v = call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %x, i1 true)
  ret <16 x i8> %v
}

; CHECK-LABEL: cttz_v16i8:
; CHECK: i32.ctz
declare <16 x i8> @llvm.cttz.v16i8(<16 x i8>, i1)
define <16 x i8> @cttz_v16i8(<16 x i8> %x) {
  %v = call <16 x i8> @llvm.cttz.v16i8(<16 x i8> %x, i1 false)
  ret <16 x i8> %v
}

; CHECK-LABEL: cttz_v16i8_undef:
; CHECK: i32.ctz
define <16 x i8> @cttz_v16i8_undef(<16 x i8> %x) {
  %v = call <16 x i8> @llvm.cttz.v16i8(<16 x i8> %x, i1 true)
  ret <16 x i8> %v
}

; CHECK-LABEL: ctpop_v16i8:
; Note: expansion does not use i32.popcnt
; CHECK: v128.and
declare <16 x i8> @llvm.ctpop.v16i8(<16 x i8>)
define <16 x i8> @ctpop_v16i8(<16 x i8> %x) {
  %v = call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %x)
  ret <16 x i8> %v
}

; CHECK-LABEL: sdiv_v16i8:
; CHECK: i32.div_s
define <16 x i8> @sdiv_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %v = sdiv <16 x i8> %x, %y
  ret <16 x i8> %v
}

; CHECK-LABEL: udiv_v16i8:
; CHECK: i32.div_u
define <16 x i8> @udiv_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %v = udiv <16 x i8> %x, %y
  ret <16 x i8> %v
}

; CHECK-LABEL: srem_v16i8:
; CHECK: i32.rem_s
define <16 x i8> @srem_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %v = srem <16 x i8> %x, %y
  ret <16 x i8> %v
}

; CHECK-LABEL: urem_v16i8:
; CHECK: i32.rem_u
define <16 x i8> @urem_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %v = urem <16 x i8> %x, %y
  ret <16 x i8> %v
}

; CHECK-LABEL: rotl_v16i8:
; Note: expansion does not use i32.rotl
; CHECK: i32.shl
declare <16 x i8> @llvm.fshl.v16i8(<16 x i8>, <16 x i8>, <16 x i8>)
define <16 x i8> @rotl_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %v = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %x, <16 x i8> %x, <16 x i8> %y)
  ret <16 x i8> %v
}

; CHECK-LABEL: rotr_v16i8:
; Note: expansion does not use i32.rotr
; CHECK: i32.shr_u
declare <16 x i8> @llvm.fshr.v16i8(<16 x i8>, <16 x i8>, <16 x i8>)
define <16 x i8> @rotr_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %v = call <16 x i8> @llvm.fshr.v16i8(<16 x i8> %x, <16 x i8> %x, <16 x i8> %y)
  ret <16 x i8> %v
}

; ==============================================================================
; 8 x i16
; ==============================================================================

; CHECK-LABEL: ctlz_v8i16:
; CHECK: i32.clz
declare <8 x i16> @llvm.ctlz.v8i16(<8 x i16>, i1)
define <8 x i16> @ctlz_v8i16(<8 x i16> %x) {
  %v = call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %x, i1 false)
  ret <8 x i16> %v
}

; CHECK-LABEL: ctlz_v8i16_undef:
; CHECK: i32.clz
define <8 x i16> @ctlz_v8i16_undef(<8 x i16> %x) {
  %v = call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %x, i1 true)
  ret <8 x i16> %v
}

; CHECK-LABEL: cttz_v8i16:
; CHECK: i32.ctz
declare <8 x i16> @llvm.cttz.v8i16(<8 x i16>, i1)
define <8 x i16> @cttz_v8i16(<8 x i16> %x) {
  %v = call <8 x i16> @llvm.cttz.v8i16(<8 x i16> %x, i1 false)
  ret <8 x i16> %v
}

; CHECK-LABEL: cttz_v8i16_undef:
; CHECK: i32.ctz
define <8 x i16> @cttz_v8i16_undef(<8 x i16> %x) {
  %v = call <8 x i16> @llvm.cttz.v8i16(<8 x i16> %x, i1 true)
  ret <8 x i16> %v
}

; CHECK-LABEL: ctpop_v8i16:
; Note: expansion does not use i32.popcnt
; CHECK: v128.and
declare <8 x i16> @llvm.ctpop.v8i16(<8 x i16>)
define <8 x i16> @ctpop_v8i16(<8 x i16> %x) {
  %v = call <8 x i16> @llvm.ctpop.v8i16(<8 x i16> %x)
  ret <8 x i16> %v
}

; CHECK-LABEL: sdiv_v8i16:
; CHECK: i32.div_s
define <8 x i16> @sdiv_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %v = sdiv <8 x i16> %x, %y
  ret <8 x i16> %v
}

; CHECK-LABEL: udiv_v8i16:
; CHECK: i32.div_u
define <8 x i16> @udiv_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %v = udiv <8 x i16> %x, %y
  ret <8 x i16> %v
}

; CHECK-LABEL: srem_v8i16:
; CHECK: i32.rem_s
define <8 x i16> @srem_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %v = srem <8 x i16> %x, %y
  ret <8 x i16> %v
}

; CHECK-LABEL: urem_v8i16:
; CHECK: i32.rem_u
define <8 x i16> @urem_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %v = urem <8 x i16> %x, %y
  ret <8 x i16> %v
}

; CHECK-LABEL: rotl_v8i16:
; Note: expansion does not use i32.rotl
; CHECK: i32.shl
declare <8 x i16> @llvm.fshl.v8i16(<8 x i16>, <8 x i16>, <8 x i16>)
define <8 x i16> @rotl_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %v = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %x, <8 x i16> %x, <8 x i16> %y)
  ret <8 x i16> %v
}

; CHECK-LABEL: rotr_v8i16:
; Note: expansion does not use i32.rotr
; CHECK: i32.shr_u
declare <8 x i16> @llvm.fshr.v8i16(<8 x i16>, <8 x i16>, <8 x i16>)
define <8 x i16> @rotr_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %v = call <8 x i16> @llvm.fshr.v8i16(<8 x i16> %x, <8 x i16> %x, <8 x i16> %y)
  ret <8 x i16> %v
}

; ==============================================================================
; 4 x i32
; ==============================================================================

; CHECK-LABEL: ctlz_v4i32:
; CHECK: i32.clz
declare <4 x i32> @llvm.ctlz.v4i32(<4 x i32>, i1)
define <4 x i32> @ctlz_v4i32(<4 x i32> %x) {
  %v = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %x, i1 false)
  ret <4 x i32> %v
}

; CHECK-LABEL: ctlz_v4i32_undef:
; CHECK: i32.clz
define <4 x i32> @ctlz_v4i32_undef(<4 x i32> %x) {
  %v = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %x, i1 true)
  ret <4 x i32> %v
}

; CHECK-LABEL: cttz_v4i32:
; CHECK: i32.ctz
declare <4 x i32> @llvm.cttz.v4i32(<4 x i32>, i1)
define <4 x i32> @cttz_v4i32(<4 x i32> %x) {
  %v = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %x, i1 false)
  ret <4 x i32> %v
}

; CHECK-LABEL: cttz_v4i32_undef:
; CHECK: i32.ctz
define <4 x i32> @cttz_v4i32_undef(<4 x i32> %x) {
  %v = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %x, i1 true)
  ret <4 x i32> %v
}

; CHECK-LABEL: ctpop_v4i32:
; Note: expansion does not use i32.popcnt
; CHECK: v128.and
declare <4 x i32> @llvm.ctpop.v4i32(<4 x i32>)
define <4 x i32> @ctpop_v4i32(<4 x i32> %x) {
  %v = call <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %x)
  ret <4 x i32> %v
}

; CHECK-LABEL: sdiv_v4i32:
; CHECK: i32.div_s
define <4 x i32> @sdiv_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %v = sdiv <4 x i32> %x, %y
  ret <4 x i32> %v
}

; CHECK-LABEL: udiv_v4i32:
; CHECK: i32.div_u
define <4 x i32> @udiv_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %v = udiv <4 x i32> %x, %y
  ret <4 x i32> %v
}

; CHECK-LABEL: srem_v4i32:
; CHECK: i32.rem_s
define <4 x i32> @srem_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %v = srem <4 x i32> %x, %y
  ret <4 x i32> %v
}

; CHECK-LABEL: urem_v4i32:
; CHECK: i32.rem_u
define <4 x i32> @urem_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %v = urem <4 x i32> %x, %y
  ret <4 x i32> %v
}

; CHECK-LABEL: rotl_v4i32:
; Note: expansion does not use i32.rotl
; CHECK: i32.shl
declare <4 x i32> @llvm.fshl.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)
define <4 x i32> @rotl_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %v = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %x, <4 x i32> %x, <4 x i32> %y)
  ret <4 x i32> %v
}

; CHECK-LABEL: rotr_v4i32:
; Note: expansion does not use i32.rotr
; CHECK: i32.shr_u
declare <4 x i32> @llvm.fshr.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)
define <4 x i32> @rotr_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %v = call <4 x i32> @llvm.fshr.v4i32(<4 x i32> %x, <4 x i32> %x, <4 x i32> %y)
  ret <4 x i32> %v
}

; ==============================================================================
; 2 x i64
; ==============================================================================

; CHECK-LABEL: ctlz_v2i64:
; CHECK: i64.clz
declare <2 x i64> @llvm.ctlz.v2i64(<2 x i64>, i1)
define <2 x i64> @ctlz_v2i64(<2 x i64> %x) {
  %v = call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %x, i1 false)
  ret <2 x i64> %v
}

; CHECK-LABEL: ctlz_v2i64_undef:
; CHECK: i64.clz
define <2 x i64> @ctlz_v2i64_undef(<2 x i64> %x) {
  %v = call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %x, i1 true)
  ret <2 x i64> %v
}

; CHECK-LABEL: cttz_v2i64:
; CHECK: i64.ctz
declare <2 x i64> @llvm.cttz.v2i64(<2 x i64>, i1)
define <2 x i64> @cttz_v2i64(<2 x i64> %x) {
  %v = call <2 x i64> @llvm.cttz.v2i64(<2 x i64> %x, i1 false)
  ret <2 x i64> %v
}

; CHECK-LABEL: cttz_v2i64_undef:
; CHECK: i64.ctz
define <2 x i64> @cttz_v2i64_undef(<2 x i64> %x) {
  %v = call <2 x i64> @llvm.cttz.v2i64(<2 x i64> %x, i1 true)
  ret <2 x i64> %v
}

; CHECK-LABEL: ctpop_v2i64:
; CHECK: i64.popcnt
declare <2 x i64> @llvm.ctpop.v2i64(<2 x i64>)
define <2 x i64> @ctpop_v2i64(<2 x i64> %x) {
  %v = call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %x)
  ret <2 x i64> %v
}

; CHECK-LABEL: sdiv_v2i64:
; CHECK: i64.div_s
define <2 x i64> @sdiv_v2i64(<2 x i64> %x, <2 x i64> %y) {
  %v = sdiv <2 x i64> %x, %y
  ret <2 x i64> %v
}

; CHECK-LABEL: udiv_v2i64:
; CHECK: i64.div_u
define <2 x i64> @udiv_v2i64(<2 x i64> %x, <2 x i64> %y) {
  %v = udiv <2 x i64> %x, %y
  ret <2 x i64> %v
}

; CHECK-LABEL: srem_v2i64:
; CHECK: i64.rem_s
define <2 x i64> @srem_v2i64(<2 x i64> %x, <2 x i64> %y) {
  %v = srem <2 x i64> %x, %y
  ret <2 x i64> %v
}

; CHECK-LABEL: urem_v2i64:
; CHECK: i64.rem_u
define <2 x i64> @urem_v2i64(<2 x i64> %x, <2 x i64> %y) {
  %v = urem <2 x i64> %x, %y
  ret <2 x i64> %v
}

; CHECK-LABEL: rotl_v2i64:
; Note: expansion does not use i64.rotl
; CHECK: i64.shl
declare <2 x i64> @llvm.fshl.v2i64(<2 x i64>, <2 x i64>, <2 x i64>)
define <2 x i64> @rotl_v2i64(<2 x i64> %x, <2 x i64> %y) {
  %v = call <2 x i64> @llvm.fshl.v2i64(<2 x i64> %x, <2 x i64> %x, <2 x i64> %y)
  ret <2 x i64> %v
}

; CHECK-LABEL: rotr_v2i64:
; Note: expansion does not use i64.rotr
; CHECK: i64.shr_u
declare <2 x i64> @llvm.fshr.v2i64(<2 x i64>, <2 x i64>, <2 x i64>)
define <2 x i64> @rotr_v2i64(<2 x i64> %x, <2 x i64> %y) {
  %v = call <2 x i64> @llvm.fshr.v2i64(<2 x i64> %x, <2 x i64> %x, <2 x i64> %y)
  ret <2 x i64> %v
}

; ==============================================================================
; 4 x f32
; ==============================================================================

; CHECK-LABEL: ceil_v4f32:
; CHECK: f32.ceil
declare <4 x float> @llvm.ceil.v4f32(<4 x float>)
define <4 x float> @ceil_v4f32(<4 x float> %x) {
  %v = call <4 x float> @llvm.ceil.v4f32(<4 x float> %x)
  ret <4 x float> %v
}

; CHECK-LABEL: floor_v4f32:
; CHECK: f32.floor
declare <4 x float> @llvm.floor.v4f32(<4 x float>)
define <4 x float> @floor_v4f32(<4 x float> %x) {
  %v = call <4 x float> @llvm.floor.v4f32(<4 x float> %x)
  ret <4 x float> %v
}

; CHECK-LABEL: trunc_v4f32:
; CHECK: f32.trunc
declare <4 x float> @llvm.trunc.v4f32(<4 x float>)
define <4 x float> @trunc_v4f32(<4 x float> %x) {
  %v = call <4 x float> @llvm.trunc.v4f32(<4 x float> %x)
  ret <4 x float> %v
}

; CHECK-LABEL: nearbyint_v4f32:
; CHECK: f32.nearest
declare <4 x float> @llvm.nearbyint.v4f32(<4 x float>)
define <4 x float> @nearbyint_v4f32(<4 x float> %x) {
  %v = call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %x)
  ret <4 x float> %v
}

; CHECK-LABEL: copysign_v4f32:
; CHECK: f32.copysign
declare <4 x float> @llvm.copysign.v4f32(<4 x float>, <4 x float>)
define <4 x float> @copysign_v4f32(<4 x float> %x, <4 x float> %y) {
  %v = call <4 x float> @llvm.copysign.v4f32(<4 x float> %x, <4 x float> %y)
  ret <4 x float> %v
}

; CHECK-LABEL: sin_v4f32:
; CHECK: f32.call $push[[L:[0-9]+]]=, sinf
declare <4 x float> @llvm.sin.v4f32(<4 x float>)
define <4 x float> @sin_v4f32(<4 x float> %x) {
  %v = call <4 x float> @llvm.sin.v4f32(<4 x float> %x)
  ret <4 x float> %v
}

; CHECK-LABEL: cos_v4f32:
; CHECK: f32.call $push[[L:[0-9]+]]=, cosf
declare <4 x float> @llvm.cos.v4f32(<4 x float>)
define <4 x float> @cos_v4f32(<4 x float> %x) {
  %v = call <4 x float> @llvm.cos.v4f32(<4 x float> %x)
  ret <4 x float> %v
}

; CHECK-LABEL: powi_v4f32:
; CHECK: f32.call $push[[L:[0-9]+]]=, __powisf2
declare <4 x float> @llvm.powi.v4f32(<4 x float>, i32)
define <4 x float> @powi_v4f32(<4 x float> %x, i32 %y) {
  %v = call <4 x float> @llvm.powi.v4f32(<4 x float> %x, i32 %y)
  ret <4 x float> %v
}

; CHECK-LABEL: pow_v4f32:
; CHECK: f32.call $push[[L:[0-9]+]]=, powf
declare <4 x float> @llvm.pow.v4f32(<4 x float>, <4 x float>)
define <4 x float> @pow_v4f32(<4 x float> %x, <4 x float> %y) {
  %v = call <4 x float> @llvm.pow.v4f32(<4 x float> %x, <4 x float> %y)
  ret <4 x float> %v
}

; CHECK-LABEL: log_v4f32:
; CHECK: f32.call $push[[L:[0-9]+]]=, logf
declare <4 x float> @llvm.log.v4f32(<4 x float>)
define <4 x float> @log_v4f32(<4 x float> %x) {
  %v = call <4 x float> @llvm.log.v4f32(<4 x float> %x)
  ret <4 x float> %v
}

; CHECK-LABEL: log2_v4f32:
; CHECK: f32.call $push[[L:[0-9]+]]=, log2f
declare <4 x float> @llvm.log2.v4f32(<4 x float>)
define <4 x float> @log2_v4f32(<4 x float> %x) {
  %v = call <4 x float> @llvm.log2.v4f32(<4 x float> %x)
  ret <4 x float> %v
}

; CHECK-LABEL: log10_v4f32:
; CHECK: f32.call $push[[L:[0-9]+]]=, log10f
declare <4 x float> @llvm.log10.v4f32(<4 x float>)
define <4 x float> @log10_v4f32(<4 x float> %x) {
  %v = call <4 x float> @llvm.log10.v4f32(<4 x float> %x)
  ret <4 x float> %v
}

; CHECK-LABEL: exp_v4f32:
; CHECK: f32.call $push[[L:[0-9]+]]=, expf
declare <4 x float> @llvm.exp.v4f32(<4 x float>)
define <4 x float> @exp_v4f32(<4 x float> %x) {
  %v = call <4 x float> @llvm.exp.v4f32(<4 x float> %x)
  ret <4 x float> %v
}

; CHECK-LABEL: exp2_v4f32:
; CHECK: f32.call $push[[L:[0-9]+]]=, exp2f
declare <4 x float> @llvm.exp2.v4f32(<4 x float>)
define <4 x float> @exp2_v4f32(<4 x float> %x) {
  %v = call <4 x float> @llvm.exp2.v4f32(<4 x float> %x)
  ret <4 x float> %v
}

; CHECK-LABEL: rint_v4f32:
; CHECK: f32.nearest
declare <4 x float> @llvm.rint.v4f32(<4 x float>)
define <4 x float> @rint_v4f32(<4 x float> %x) {
  %v = call <4 x float> @llvm.rint.v4f32(<4 x float> %x)
  ret <4 x float> %v
}

; CHECK-LABEL: round_v4f32:
; CHECK: f32.call $push[[L:[0-9]+]]=, roundf
declare <4 x float> @llvm.round.v4f32(<4 x float>)
define <4 x float> @round_v4f32(<4 x float> %x) {
  %v = call <4 x float> @llvm.round.v4f32(<4 x float> %x)
  ret <4 x float> %v
}

; ==============================================================================
; 2 x f64
; ==============================================================================

; CHECK-LABEL: ceil_v2f64:
; CHECK: f64.ceil
declare <2 x double> @llvm.ceil.v2f64(<2 x double>)
define <2 x double> @ceil_v2f64(<2 x double> %x) {
  %v = call <2 x double> @llvm.ceil.v2f64(<2 x double> %x)
  ret <2 x double> %v
}

; CHECK-LABEL: floor_v2f64:
; CHECK: f64.floor
declare <2 x double> @llvm.floor.v2f64(<2 x double>)
define <2 x double> @floor_v2f64(<2 x double> %x) {
  %v = call <2 x double> @llvm.floor.v2f64(<2 x double> %x)
  ret <2 x double> %v
}

; CHECK-LABEL: trunc_v2f64:
; CHECK: f64.trunc
declare <2 x double> @llvm.trunc.v2f64(<2 x double>)
define <2 x double> @trunc_v2f64(<2 x double> %x) {
  %v = call <2 x double> @llvm.trunc.v2f64(<2 x double> %x)
  ret <2 x double> %v
}

; CHECK-LABEL: nearbyint_v2f64:
; CHECK: f64.nearest
declare <2 x double> @llvm.nearbyint.v2f64(<2 x double>)
define <2 x double> @nearbyint_v2f64(<2 x double> %x) {
  %v = call <2 x double> @llvm.nearbyint.v2f64(<2 x double> %x)
  ret <2 x double> %v
}

; CHECK-LABEL: copysign_v2f64:
; CHECK: f64.copysign
declare <2 x double> @llvm.copysign.v2f64(<2 x double>, <2 x double>)
define <2 x double> @copysign_v2f64(<2 x double> %x, <2 x double> %y) {
  %v = call <2 x double> @llvm.copysign.v2f64(<2 x double> %x, <2 x double> %y)
  ret <2 x double> %v
}

; CHECK-LABEL: sin_v2f64:
; CHECK: f64.call $push[[L:[0-9]+]]=, sin
declare <2 x double> @llvm.sin.v2f64(<2 x double>)
define <2 x double> @sin_v2f64(<2 x double> %x) {
  %v = call <2 x double> @llvm.sin.v2f64(<2 x double> %x)
  ret <2 x double> %v
}

; CHECK-LABEL: cos_v2f64:
; CHECK: f64.call $push[[L:[0-9]+]]=, cos
declare <2 x double> @llvm.cos.v2f64(<2 x double>)
define <2 x double> @cos_v2f64(<2 x double> %x) {
  %v = call <2 x double> @llvm.cos.v2f64(<2 x double> %x)
  ret <2 x double> %v
}

; CHECK-LABEL: powi_v2f64:
; CHECK: f64.call $push[[L:[0-9]+]]=, __powidf2
declare <2 x double> @llvm.powi.v2f64(<2 x double>, i32)
define <2 x double> @powi_v2f64(<2 x double> %x, i32 %y) {
  %v = call <2 x double> @llvm.powi.v2f64(<2 x double> %x, i32 %y)
  ret <2 x double> %v
}

; CHECK-LABEL: pow_v2f64:
; CHECK: f64.call $push[[L:[0-9]+]]=, pow
declare <2 x double> @llvm.pow.v2f64(<2 x double>, <2 x double>)
define <2 x double> @pow_v2f64(<2 x double> %x, <2 x double> %y) {
  %v = call <2 x double> @llvm.pow.v2f64(<2 x double> %x, <2 x double> %y)
  ret <2 x double> %v
}

; CHECK-LABEL: log_v2f64:
; CHECK: f64.call $push[[L:[0-9]+]]=, log
declare <2 x double> @llvm.log.v2f64(<2 x double>)
define <2 x double> @log_v2f64(<2 x double> %x) {
  %v = call <2 x double> @llvm.log.v2f64(<2 x double> %x)
  ret <2 x double> %v
}

; CHECK-LABEL: log2_v2f64:
; CHECK: f64.call $push[[L:[0-9]+]]=, log2
declare <2 x double> @llvm.log2.v2f64(<2 x double>)
define <2 x double> @log2_v2f64(<2 x double> %x) {
  %v = call <2 x double> @llvm.log2.v2f64(<2 x double> %x)
  ret <2 x double> %v
}

; CHECK-LABEL: log10_v2f64:
; CHECK: f64.call $push[[L:[0-9]+]]=, log10
declare <2 x double> @llvm.log10.v2f64(<2 x double>)
define <2 x double> @log10_v2f64(<2 x double> %x) {
  %v = call <2 x double> @llvm.log10.v2f64(<2 x double> %x)
  ret <2 x double> %v
}

; CHECK-LABEL: exp_v2f64:
; CHECK: f64.call $push[[L:[0-9]+]]=, exp
declare <2 x double> @llvm.exp.v2f64(<2 x double>)
define <2 x double> @exp_v2f64(<2 x double> %x) {
  %v = call <2 x double> @llvm.exp.v2f64(<2 x double> %x)
  ret <2 x double> %v
}

; CHECK-LABEL: exp2_v2f64:
; CHECK: f64.call $push[[L:[0-9]+]]=, exp2
declare <2 x double> @llvm.exp2.v2f64(<2 x double>)
define <2 x double> @exp2_v2f64(<2 x double> %x) {
  %v = call <2 x double> @llvm.exp2.v2f64(<2 x double> %x)
  ret <2 x double> %v
}

; CHECK-LABEL: rint_v2f64:
; CHECK: f64.nearest
declare <2 x double> @llvm.rint.v2f64(<2 x double>)
define <2 x double> @rint_v2f64(<2 x double> %x) {
  %v = call <2 x double> @llvm.rint.v2f64(<2 x double> %x)
  ret <2 x double> %v
}

; CHECK-LABEL: round_v2f64:
; CHECK: f64.call $push[[L:[0-9]+]]=, round
declare <2 x double> @llvm.round.v2f64(<2 x double>)
define <2 x double> @round_v2f64(<2 x double> %x) {
  %v = call <2 x double> @llvm.round.v2f64(<2 x double> %x)
  ret <2 x double> %v
}
