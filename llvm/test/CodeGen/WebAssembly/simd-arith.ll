; RUN: llc < %s -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals -wasm-enable-unimplemented-simd -mattr=+simd128 --show-mc-encoding | FileCheck %s --check-prefixes CHECK,SIMD128
; RUN: llc < %s -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals -wasm-enable-unimplemented-simd -mattr=+simd128 -fast-isel --show-mc-encoding | FileCheck %s --check-prefixes CHECK,SIMD128
; RUN: llc < %s -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals -mattr=+simd128 --show-mc-encoding | FileCheck %s --check-prefixes CHECK,SIMD128-VM
; RUN: llc < %s -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals -mattr=+simd128 -fast-isel --show-mc-encoding | FileCheck %s --check-prefixes CHECK,SIMD128-VM
; RUN: llc < %s -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals -mattr=-simd128 --show-mc-encoding | FileCheck %s --check-prefixes CHECK,NO-SIMD128
; RUN: llc < %s -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals -mattr=-simd128 -fast-isel --show-mc-encoding | FileCheck %s --check-prefixes CHECK,NO-SIMD128

; Test that basic SIMD128 arithmetic operations assemble as expected.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; ==============================================================================
; 16 x i8
; ==============================================================================
; CHECK-LABEL: add_v16i8
; NO-SIMD128-NOT: i8x16
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i8x16.add $push0=, $0, $1 # encoding: [0xfd,0x18]{{$}}
; SIMD128: return $pop0 #
define <16 x i8> @add_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = add <16 x i8> %x, %y
  ret <16 x i8> %a
}

; CHECK-LABEL: sub_v16i8
; NO-SIMD128-NOT: i8x16
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i8x16.sub $push0=, $0, $1 # encoding: [0xfd,0x1c]{{$}}
; SIMD128: return $pop0 #
define <16 x i8> @sub_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = sub <16 x i8> %x, %y
  ret <16 x i8> %a
}

; CHECK-LABEL: mul_v16i8
; NO-SIMD128-NOT: i8x16
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i8x16.mul $push0=, $0, $1 # encoding: [0xfd,0x20]{{$}}
; SIMD128: return $pop0 #
define <16 x i8> @mul_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = mul <16 x i8> %x, %y
  ret <16 x i8> %a
}

; ==============================================================================
; 8 x i16
; ==============================================================================
; CHECK-LABEL: add_v8i16
; NO-SIMD128-NOT: i16x8
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i16x8.add $push0=, $0, $1 # encoding: [0xfd,0x19]{{$}}
; SIMD128: return $pop0 #
define <8 x i16> @add_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = add <8 x i16> %x, %y
  ret <8 x i16> %a
}

; CHECK-LABEL: sub_v8i16
; NO-SIMD128-NOT: i16x8
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i16x8.sub $push0=, $0, $1 # encoding: [0xfd,0x1d]{{$}}
; SIMD128: return $pop0 #
define <8 x i16> @sub_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = sub <8 x i16> %x, %y
  ret <8 x i16> %a
}

; CHECK-LABEL: mul_v8i16
; NO-SIMD128-NOT: i16x8
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i16x8.mul $push0=, $0, $1 # encoding: [0xfd,0x21]{{$}}
; SIMD128: return $pop0 #
define <8 x i16> @mul_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = mul <8 x i16> %x, %y
  ret <8 x i16> %a
}

; ==============================================================================
; 4 x i32
; ==============================================================================
; CHECK-LABEL: add_v4i32
; NO-SIMD128-NOT: i32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i32x4.add $push0=, $0, $1 # encoding: [0xfd,0x1a]{{$}}
; SIMD128: return $pop0 #
define <4 x i32> @add_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %a = add <4 x i32> %x, %y
  ret <4 x i32> %a
}

; CHECK-LABEL: sub_v4i32
; NO-SIMD128-NOT: i32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i32x4.sub $push0=, $0, $1 # encoding: [0xfd,0x1e]{{$}}
; SIMD128: return $pop0 #
define <4 x i32> @sub_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %a = sub <4 x i32> %x, %y
  ret <4 x i32> %a
}

; CHECK-LABEL: mul_v4i32
; NO-SIMD128-NOT: i32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i32x4.mul $push0=, $0, $1 # encoding: [0xfd,0x22]{{$}}
; SIMD128: return $pop0 #
define <4 x i32> @mul_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %a = mul <4 x i32> %x, %y
  ret <4 x i32> %a
}

; ==============================================================================
; 2 x i64
; ==============================================================================
; CHECK-LABEL: add_v2i64
; NO-SIMD128-NOT: i64x2
; SIMD128-VM-NOT: i64x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i64x2.add $push0=, $0, $1 # encoding: [0xfd,0x1b]{{$}}
; SIMD128: return $pop0 #
define <2 x i64> @add_v2i64(<2 x i64> %x, <2 x i64> %y) {
  %a = add <2 x i64> %x, %y
  ret <2 x i64> %a
}

; CHECK-LABEL: sub_v2i64
; NO-SIMD128-NOT: i64x2
; SIMD128-VM-NOT: i64x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i64x2.sub $push0=, $0, $1 # encoding: [0xfd,0x1f]{{$}}
; SIMD128: return $pop0 #
define <2 x i64> @sub_v2i64(<2 x i64> %x, <2 x i64> %y) {
  %a = sub <2 x i64> %x, %y
  ret <2 x i64> %a
}

; CHECK-LABEL: mul_v2i64
; NO-SIMD128-NOT: i64x2
; SIMD128-VM-NOT: i64x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i64x2.mul $push0=, $0, $1 # encoding: [0xfd,0x23]{{$}}
; SIMD128: return $pop0 #
define <2 x i64> @mul_v2i64(<2 x i64> %x, <2 x i64> %y) {
  %a = mul <2 x i64> %x, %y
  ret <2 x i64> %a
}

; ==============================================================================
; 4 x float
; ==============================================================================
; CHECK-LABEL: add_v4f32
; NO-SIMD128-NOT: f32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f32x4.add $push0=, $0, $1 # encoding: [0xfd,0x7a]{{$}}
; SIMD128: return $pop0 #
define <4 x float> @add_v4f32(<4 x float> %x, <4 x float> %y) {
  %a = fadd <4 x float> %x, %y
  ret <4 x float> %a
}

; CHECK-LABEL: sub_v4f32
; NO-SIMD128-NOT: f32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f32x4.sub $push0=, $0, $1 # encoding: [0xfd,0x7c]{{$}}
; SIMD128: return $pop0 #
define <4 x float> @sub_v4f32(<4 x float> %x, <4 x float> %y) {
  %a = fsub <4 x float> %x, %y
  ret <4 x float> %a
}

; CHECK-LABEL: div_v4f32
; NO-SIMD128-NOT: f32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f32x4.div $push0=, $0, $1 # encoding: [0xfd,0x7e]{{$}}
; SIMD128: return $pop0 #
define <4 x float> @div_v4f32(<4 x float> %x, <4 x float> %y) {
  %a = fdiv <4 x float> %x, %y
  ret <4 x float> %a
}

; CHECK-LABEL: mul_v4f32
; NO-SIMD128-NOT: f32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f32x4.mul $push0=, $0, $1 # encoding: [0xfd,0x80]{{$}}
; SIMD128: return $pop0 #
define <4 x float> @mul_v4f32(<4 x float> %x, <4 x float> %y) {
  %a = fmul <4 x float> %x, %y
  ret <4 x float> %a
}

; ==============================================================================
; 2 x double
; ==============================================================================
; CHECK-LABEL: add_v2f64
; NO-SIMD128-NOT: f64x2
; SIMD129-VM-NOT: f62x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f64x2.add $push0=, $0, $1 # encoding: [0xfd,0x7b]{{$}}
; SIMD128: return $pop0 #
define <2 x double> @add_v2f64(<2 x double> %x, <2 x double> %y) {
  %a = fadd <2 x double> %x, %y
  ret <2 x double> %a
}

; CHECK-LABEL: sub_v2f64
; NO-SIMD128-NOT: f64x2
; SIMD129-VM-NOT: f62x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f64x2.sub $push0=, $0, $1 # encoding: [0xfd,0x7d]{{$}}
; SIMD128: return $pop0 #
define <2 x double> @sub_v2f64(<2 x double> %x, <2 x double> %y) {
  %a = fsub <2 x double> %x, %y
  ret <2 x double> %a
}

; CHECK-LABEL: div_v2f64
; NO-SIMD128-NOT: f64x2
; SIMD129-VM-NOT: f62x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f64x2.div $push0=, $0, $1 # encoding: [0xfd,0x7f]{{$}}
; SIMD128: return $pop0 #
define <2 x double> @div_v2f64(<2 x double> %x, <2 x double> %y) {
  %a = fdiv <2 x double> %x, %y
  ret <2 x double> %a
}

; CHECK-LABEL: mul_v2f64
; NO-SIMD128-NOT: f64x2
; SIMD129-VM-NOT: f62x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f64x2.mul $push0=, $0, $1 # encoding: [0xfd,0x81]{{$}}
; SIMD128: return $pop0 #
define <2 x double> @mul_v2f64(<2 x double> %x, <2 x double> %y) {
  %a = fmul <2 x double> %x, %y
  ret <2 x double> %a
}
