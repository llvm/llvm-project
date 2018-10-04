; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -wasm-enable-unimplemented-simd -mattr=+simd128 | FileCheck %s --check-prefixes CHECK,SIMD128
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -wasm-enable-unimplemented-simd -mattr=+simd128 -fast-isel | FileCheck %s --check-prefixes CHECK,SIMD128

; Test that SIMD128 intrinsics lower as expected. These intrinsics are
; only expected to lower successfully if the simd128 attribute is
; enabled and legal types are used.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; ==============================================================================
; 16 x i8
; ==============================================================================
; CHECK-LABEL: any_v16i8:
; SIMD128-NEXT: .param v128{{$}}
; SIMD128-NEXT: .result i32{{$}}
; SIMD128-NEXT: i8x16.any_true $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.anytrue.v16i8(<16 x i8>)
define i32 @any_v16i8(<16 x i8> %x) {
  %a = call i32 @llvm.wasm.anytrue.v16i8(<16 x i8> %x)
  ret i32 %a
}

; CHECK-LABEL: all_v16i8:
; SIMD128-NEXT: .param v128{{$}}
; SIMD128-NEXT: .result i32{{$}}
; SIMD128-NEXT: i8x16.all_true $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.alltrue.v16i8(<16 x i8>)
define i32 @all_v16i8(<16 x i8> %x) {
  %a = call i32 @llvm.wasm.alltrue.v16i8(<16 x i8> %x)
  ret i32 %a
}

; CHECK-LABEL: bitselect_v16i8:
; SIMD128-NEXT: .param v128, v128, v128{{$}}
; SIMD128-NEXT: .result v128{{$}}
; SIMD128-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.wasm.bitselect.v16i8(<16 x i8>, <16 x i8>, <16 x i8>)
define <16 x i8> @bitselect_v16i8(<16 x i8> %c, <16 x i8> %v1, <16 x i8> %v2) {
  %a = call <16 x i8> @llvm.wasm.bitselect.v16i8(
    <16 x i8> %c, <16 x i8> %v1, <16 x i8> %v2
  )
  ret <16 x i8> %a
}

; ==============================================================================
; 8 x i16
; ==============================================================================
; CHECK-LABEL: any_v8i16:
; SIMD128-NEXT: .param v128{{$}}
; SIMD128-NEXT: .result i32{{$}}
; SIMD128-NEXT: i16x8.any_true $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.anytrue.v8i16(<8 x i16>)
define i32 @any_v8i16(<8 x i16> %x) {
  %a = call i32 @llvm.wasm.anytrue.v8i16(<8 x i16> %x)
  ret i32 %a
}

; CHECK-LABEL: all_v8i16:
; SIMD128-NEXT: .param v128{{$}}
; SIMD128-NEXT: .result i32{{$}}
; SIMD128-NEXT: i16x8.all_true $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.alltrue.v8i16(<8 x i16>)
define i32 @all_v8i16(<8 x i16> %x) {
  %a = call i32 @llvm.wasm.alltrue.v8i16(<8 x i16> %x)
  ret i32 %a
}

; CHECK-LABEL: bitselect_v8i16:
; SIMD128-NEXT: .param v128, v128, v128{{$}}
; SIMD128-NEXT: .result v128{{$}}
; SIMD128-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.bitselect.v8i16(<8 x i16>, <8 x i16>, <8 x i16>)
define <8 x i16> @bitselect_v8i16(<8 x i16> %c, <8 x i16> %v1, <8 x i16> %v2) {
  %a = call <8 x i16> @llvm.wasm.bitselect.v8i16(
    <8 x i16> %c, <8 x i16> %v1, <8 x i16> %v2
  )
  ret <8 x i16> %a
}

; ==============================================================================
; 4 x i32
; ==============================================================================
; CHECK-LABEL: any_v4i32:
; SIMD128-NEXT: .param v128{{$}}
; SIMD128-NEXT: .result i32{{$}}
; SIMD128-NEXT: i32x4.any_true $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.anytrue.v4i32(<4 x i32>)
define i32 @any_v4i32(<4 x i32> %x) {
  %a = call i32 @llvm.wasm.anytrue.v4i32(<4 x i32> %x)
  ret i32 %a
}

; CHECK-LABEL: all_v4i32:
; SIMD128-NEXT: .param v128{{$}}
; SIMD128-NEXT: .result i32{{$}}
; SIMD128-NEXT: i32x4.all_true $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.alltrue.v4i32(<4 x i32>)
define i32 @all_v4i32(<4 x i32> %x) {
  %a = call i32 @llvm.wasm.alltrue.v4i32(<4 x i32> %x)
  ret i32 %a
}

; CHECK-LABEL: bitselect_v4i32:
; SIMD128-NEXT: .param v128, v128, v128{{$}}
; SIMD128-NEXT: .result v128{{$}}
; SIMD128-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <4 x i32> @llvm.wasm.bitselect.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)
define <4 x i32> @bitselect_v4i32(<4 x i32> %c, <4 x i32> %v1, <4 x i32> %v2) {
  %a = call <4 x i32> @llvm.wasm.bitselect.v4i32(
    <4 x i32> %c, <4 x i32> %v1, <4 x i32> %v2
  )
  ret <4 x i32> %a
}

; ==============================================================================
; 2 x i64
; ==============================================================================
; CHECK-LABEL: any_v2i64:
; SIMD128-NEXT: .param v128{{$}}
; SIMD128-NEXT: .result i32{{$}}
; SIMD128-NEXT: i64x2.any_true $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.anytrue.v2i64(<2 x i64>)
define i32 @any_v2i64(<2 x i64> %x) {
  %a = call i32 @llvm.wasm.anytrue.v2i64(<2 x i64> %x)
  ret i32 %a
}

; CHECK-LABEL: all_v2i64:
; SIMD128-NEXT: .param v128{{$}}
; SIMD128-NEXT: .result i32{{$}}
; SIMD128-NEXT: i64x2.all_true $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.alltrue.v2i64(<2 x i64>)
define i32 @all_v2i64(<2 x i64> %x) {
  %a = call i32 @llvm.wasm.alltrue.v2i64(<2 x i64> %x)
  ret i32 %a
}

; CHECK-LABEL: bitselect_v2i64:
; SIMD128-NEXT: .param v128, v128, v128{{$}}
; SIMD128-NEXT: .result v128{{$}}
; SIMD128-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <2 x i64> @llvm.wasm.bitselect.v2i64(<2 x i64>, <2 x i64>, <2 x i64>)
define <2 x i64> @bitselect_v2i64(<2 x i64> %c, <2 x i64> %v1, <2 x i64> %v2) {
  %a = call <2 x i64> @llvm.wasm.bitselect.v2i64(
    <2 x i64> %c, <2 x i64> %v1, <2 x i64> %v2
  )
  ret <2 x i64> %a
}

; ==============================================================================
; 4 x f32
; ==============================================================================
; CHECK-LABEL: bitselect_v4f32:
; SIMD128-NEXT: .param v128, v128, v128{{$}}
; SIMD128-NEXT: .result v128{{$}}
; SIMD128-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <4 x float> @llvm.wasm.bitselect.v4f32(<4 x float>, <4 x float>, <4 x float>)
define <4 x float> @bitselect_v4f32(<4 x float> %c, <4 x float> %v1, <4 x float> %v2) {
  %a = call <4 x float> @llvm.wasm.bitselect.v4f32(
    <4 x float> %c, <4 x float> %v1, <4 x float> %v2
  )
  ret <4 x float> %a
}

; ==============================================================================
; 2 x f64
; ==============================================================================
; CHECK-LABEL: bitselect_v2f64:
; SIMD128-NEXT: .param v128, v128, v128{{$}}
; SIMD128-NEXT: .result v128{{$}}
; SIMD128-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <2 x double> @llvm.wasm.bitselect.v2f64(<2 x double>, <2 x double>, <2 x double>)
define <2 x double> @bitselect_v2f64(<2 x double> %c, <2 x double> %v1, <2 x double> %v2) {
  %a = call <2 x double> @llvm.wasm.bitselect.v2f64(
    <2 x double> %c, <2 x double> %v1, <2 x double> %v2
  )
  ret <2 x double> %a
}
