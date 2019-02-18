; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -wasm-enable-unimplemented-simd -mattr=+simd128 | FileCheck %s

; Test loads and stores with custom alignment values.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; ==============================================================================
; 16 x i8
; ==============================================================================

; CHECK-LABEL: load_v16i8_a1:
; CHECK-NEXT: .functype load_v16i8_a1 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_v16i8_a1(<16 x i8> *%p) {
  %v = load <16 x i8>, <16 x i8>* %p, align 1
  ret <16 x i8> %v
}

; CHECK-LABEL: load_v16i8_a4:
; CHECK-NEXT: .functype load_v16i8_a4 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0):p2align=2{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_v16i8_a4(<16 x i8> *%p) {
  %v = load <16 x i8>, <16 x i8>* %p, align 4
  ret <16 x i8> %v
}

; 16 is the default alignment for v128 so no attribute is needed.

; CHECK-LABEL: load_v16i8_a16:
; CHECK-NEXT: .functype load_v16i8_a16 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_v16i8_a16(<16 x i8> *%p) {
  %v = load <16 x i8>, <16 x i8>* %p, align 16
  ret <16 x i8> %v
}

; 32 is greater than the default alignment so it is ignored.

; CHECK-LABEL: load_v16i8_a32:
; CHECK-NEXT: .functype load_v16i8_a32 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_v16i8_a32(<16 x i8> *%p) {
  %v = load <16 x i8>, <16 x i8>* %p, align 32
  ret <16 x i8> %v
}

; CHECK-LABEL: store_v16i8_a1:
; CHECK-NEXT: .functype store_v16i8_a1 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0):p2align=0, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v16i8_a1(<16 x i8> *%p, <16 x i8> %v) {
  store <16 x i8> %v, <16 x i8>* %p, align 1
  ret void
}

; CHECK-LABEL: store_v16i8_a4:
; CHECK-NEXT: .functype store_v16i8_a4 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0):p2align=2, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v16i8_a4(<16 x i8> *%p, <16 x i8> %v) {
  store <16 x i8> %v, <16 x i8>* %p, align 4
  ret void
}

; 16 is the default alignment for v128 so no attribute is needed.

; CHECK-LABEL: store_v16i8_a16:
; CHECK-NEXT: .functype store_v16i8_a16 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v16i8_a16(<16 x i8> *%p, <16 x i8> %v) {
  store <16 x i8> %v, <16 x i8>* %p, align 16
  ret void
}

; 32 is greater than the default alignment so it is ignored.

; CHECK-LABEL: store_v16i8_a32:
; CHECK-NEXT: .functype store_v16i8_a32 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v16i8_a32(<16 x i8> *%p, <16 x i8> %v) {
  store <16 x i8> %v, <16 x i8>* %p, align 32
  ret void
}

; ==============================================================================
; 8 x i16
; ==============================================================================

; CHECK-LABEL: load_v8i16_a1:
; CHECK-NEXT: .functype load_v8i16_a1 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_v8i16_a1(<8 x i16> *%p) {
  %v = load <8 x i16>, <8 x i16>* %p, align 1
  ret <8 x i16> %v
}

; CHECK-LABEL: load_v8i16_a4:
; CHECK-NEXT: .functype load_v8i16_a4 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0):p2align=2{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_v8i16_a4(<8 x i16> *%p) {
  %v = load <8 x i16>, <8 x i16>* %p, align 4
  ret <8 x i16> %v
}

; 8 is the default alignment for v128 so no attribute is needed.

; CHECK-LABEL: load_v8i16_a16:
; CHECK-NEXT: .functype load_v8i16_a16 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_v8i16_a16(<8 x i16> *%p) {
  %v = load <8 x i16>, <8 x i16>* %p, align 16
  ret <8 x i16> %v
}

; 32 is greater than the default alignment so it is ignored.

; CHECK-LABEL: load_v8i16_a32:
; CHECK-NEXT: .functype load_v8i16_a32 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_v8i16_a32(<8 x i16> *%p) {
  %v = load <8 x i16>, <8 x i16>* %p, align 32
  ret <8 x i16> %v
}

; CHECK-LABEL: store_v8i16_a1:
; CHECK-NEXT: .functype store_v8i16_a1 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0):p2align=0, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v8i16_a1(<8 x i16> *%p, <8 x i16> %v) {
  store <8 x i16> %v, <8 x i16>* %p, align 1
  ret void
}

; CHECK-LABEL: store_v8i16_a4:
; CHECK-NEXT: .functype store_v8i16_a4 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0):p2align=2, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v8i16_a4(<8 x i16> *%p, <8 x i16> %v) {
  store <8 x i16> %v, <8 x i16>* %p, align 4
  ret void
}

; 16 is the default alignment for v128 so no attribute is needed.

; CHECK-LABEL: store_v8i16_a16:
; CHECK-NEXT: .functype store_v8i16_a16 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v8i16_a16(<8 x i16> *%p, <8 x i16> %v) {
  store <8 x i16> %v, <8 x i16>* %p, align 16
  ret void
}

; 32 is greater than the default alignment so it is ignored.

; CHECK-LABEL: store_v8i16_a32:
; CHECK-NEXT: .functype store_v8i16_a32 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v8i16_a32(<8 x i16> *%p, <8 x i16> %v) {
  store <8 x i16> %v, <8 x i16>* %p, align 32
  ret void
}

; ==============================================================================
; 4 x i32
; ==============================================================================

; CHECK-LABEL: load_v4i32_a1:
; CHECK-NEXT: .functype load_v4i32_a1 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_v4i32_a1(<4 x i32> *%p) {
  %v = load <4 x i32>, <4 x i32>* %p, align 1
  ret <4 x i32> %v
}

; CHECK-LABEL: load_v4i32_a4:
; CHECK-NEXT: .functype load_v4i32_a4 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0):p2align=2{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_v4i32_a4(<4 x i32> *%p) {
  %v = load <4 x i32>, <4 x i32>* %p, align 4
  ret <4 x i32> %v
}

; 4 is the default alignment for v128 so no attribute is needed.

; CHECK-LABEL: load_v4i32_a16:
; CHECK-NEXT: .functype load_v4i32_a16 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_v4i32_a16(<4 x i32> *%p) {
  %v = load <4 x i32>, <4 x i32>* %p, align 16
  ret <4 x i32> %v
}

; 32 is greater than the default alignment so it is ignored.

; CHECK-LABEL: load_v4i32_a32:
; CHECK-NEXT: .functype load_v4i32_a32 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_v4i32_a32(<4 x i32> *%p) {
  %v = load <4 x i32>, <4 x i32>* %p, align 32
  ret <4 x i32> %v
}

; CHECK-LABEL: store_v4i32_a1:
; CHECK-NEXT: .functype store_v4i32_a1 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0):p2align=0, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v4i32_a1(<4 x i32> *%p, <4 x i32> %v) {
  store <4 x i32> %v, <4 x i32>* %p, align 1
  ret void
}

; CHECK-LABEL: store_v4i32_a4:
; CHECK-NEXT: .functype store_v4i32_a4 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0):p2align=2, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v4i32_a4(<4 x i32> *%p, <4 x i32> %v) {
  store <4 x i32> %v, <4 x i32>* %p, align 4
  ret void
}

; 16 is the default alignment for v128 so no attribute is needed.

; CHECK-LABEL: store_v4i32_a16:
; CHECK-NEXT: .functype store_v4i32_a16 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v4i32_a16(<4 x i32> *%p, <4 x i32> %v) {
  store <4 x i32> %v, <4 x i32>* %p, align 16
  ret void
}

; 32 is greater than the default alignment so it is ignored.

; CHECK-LABEL: store_v4i32_a32:
; CHECK-NEXT: .functype store_v4i32_a32 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v4i32_a32(<4 x i32> *%p, <4 x i32> %v) {
  store <4 x i32> %v, <4 x i32>* %p, align 32
  ret void
}

; ==============================================================================
; 2 x i64
; ==============================================================================

; CHECK-LABEL: load_v2i64_a1:
; CHECK-NEXT: .functype load_v2i64_a1 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_v2i64_a1(<2 x i64> *%p) {
  %v = load <2 x i64>, <2 x i64>* %p, align 1
  ret <2 x i64> %v
}

; CHECK-LABEL: load_v2i64_a4:
; CHECK-NEXT: .functype load_v2i64_a4 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0):p2align=2{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_v2i64_a4(<2 x i64> *%p) {
  %v = load <2 x i64>, <2 x i64>* %p, align 4
  ret <2 x i64> %v
}

; 2 is the default alignment for v128 so no attribute is needed.

; CHECK-LABEL: load_v2i64_a16:
; CHECK-NEXT: .functype load_v2i64_a16 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_v2i64_a16(<2 x i64> *%p) {
  %v = load <2 x i64>, <2 x i64>* %p, align 16
  ret <2 x i64> %v
}

; 32 is greater than the default alignment so it is ignored.

; CHECK-LABEL: load_v2i64_a32:
; CHECK-NEXT: .functype load_v2i64_a32 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_v2i64_a32(<2 x i64> *%p) {
  %v = load <2 x i64>, <2 x i64>* %p, align 32
  ret <2 x i64> %v
}

; CHECK-LABEL: store_v2i64_a1:
; CHECK-NEXT: .functype store_v2i64_a1 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0):p2align=0, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v2i64_a1(<2 x i64> *%p, <2 x i64> %v) {
  store <2 x i64> %v, <2 x i64>* %p, align 1
  ret void
}

; CHECK-LABEL: store_v2i64_a4:
; CHECK-NEXT: .functype store_v2i64_a4 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0):p2align=2, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v2i64_a4(<2 x i64> *%p, <2 x i64> %v) {
  store <2 x i64> %v, <2 x i64>* %p, align 4
  ret void
}

; 16 is the default alignment for v128 so no attribute is needed.

; CHECK-LABEL: store_v2i64_a16:
; CHECK-NEXT: .functype store_v2i64_a16 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v2i64_a16(<2 x i64> *%p, <2 x i64> %v) {
  store <2 x i64> %v, <2 x i64>* %p, align 16
  ret void
}

; 32 is greater than the default alignment so it is ignored.

; CHECK-LABEL: store_v2i64_a32:
; CHECK-NEXT: .functype store_v2i64_a32 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v2i64_a32(<2 x i64> *%p, <2 x i64> %v) {
  store <2 x i64> %v, <2 x i64>* %p, align 32
  ret void
}

; ==============================================================================
; 4 x float
; ==============================================================================

; CHECK-LABEL: load_v4f32_a1:
; CHECK-NEXT: .functype load_v4f32_a1 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x float> @load_v4f32_a1(<4 x float> *%p) {
  %v = load <4 x float>, <4 x float>* %p, align 1
  ret <4 x float> %v
}

; CHECK-LABEL: load_v4f32_a4:
; CHECK-NEXT: .functype load_v4f32_a4 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0):p2align=2{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x float> @load_v4f32_a4(<4 x float> *%p) {
  %v = load <4 x float>, <4 x float>* %p, align 4
  ret <4 x float> %v
}

; 4 is the default alignment for v128 so no attribute is needed.

; CHECK-LABEL: load_v4f32_a16:
; CHECK-NEXT: .functype load_v4f32_a16 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x float> @load_v4f32_a16(<4 x float> *%p) {
  %v = load <4 x float>, <4 x float>* %p, align 16
  ret <4 x float> %v
}

; 32 is greater than the default alignment so it is ignored.

; CHECK-LABEL: load_v4f32_a32:
; CHECK-NEXT: .functype load_v4f32_a32 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x float> @load_v4f32_a32(<4 x float> *%p) {
  %v = load <4 x float>, <4 x float>* %p, align 32
  ret <4 x float> %v
}

; CHECK-LABEL: store_v4f32_a1:
; CHECK-NEXT: .functype store_v4f32_a1 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0):p2align=0, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v4f32_a1(<4 x float> *%p, <4 x float> %v) {
  store <4 x float> %v, <4 x float>* %p, align 1
  ret void
}

; CHECK-LABEL: store_v4f32_a4:
; CHECK-NEXT: .functype store_v4f32_a4 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0):p2align=2, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v4f32_a4(<4 x float> *%p, <4 x float> %v) {
  store <4 x float> %v, <4 x float>* %p, align 4
  ret void
}

; 16 is the default alignment for v128 so no attribute is needed.

; CHECK-LABEL: store_v4f32_a16:
; CHECK-NEXT: .functype store_v4f32_a16 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v4f32_a16(<4 x float> *%p, <4 x float> %v) {
  store <4 x float> %v, <4 x float>* %p, align 16
  ret void
}

; 32 is greater than the default alignment so it is ignored.

; CHECK-LABEL: store_v4f32_a32:
; CHECK-NEXT: .functype store_v4f32_a32 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v4f32_a32(<4 x float> *%p, <4 x float> %v) {
  store <4 x float> %v, <4 x float>* %p, align 32
  ret void
}

; ==============================================================================
; 2 x double
; ==============================================================================

; CHECK-LABEL: load_v2f64_a1:
; CHECK-NEXT: .functype load_v2f64_a1 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x double> @load_v2f64_a1(<2 x double> *%p) {
  %v = load <2 x double>, <2 x double>* %p, align 1
  ret <2 x double> %v
}

; CHECK-LABEL: load_v2f64_a4:
; CHECK-NEXT: .functype load_v2f64_a4 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0):p2align=2{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x double> @load_v2f64_a4(<2 x double> *%p) {
  %v = load <2 x double>, <2 x double>* %p, align 4
  ret <2 x double> %v
}

; 2 is the default alignment for v128 so no attribute is needed.

; CHECK-LABEL: load_v2f64_a16:
; CHECK-NEXT: .functype load_v2f64_a16 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x double> @load_v2f64_a16(<2 x double> *%p) {
  %v = load <2 x double>, <2 x double>* %p, align 16
  ret <2 x double> %v
}

; 32 is greater than the default alignment so it is ignored.

; CHECK-LABEL: load_v2f64_a32:
; CHECK-NEXT: .functype load_v2f64_a32 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x double> @load_v2f64_a32(<2 x double> *%p) {
  %v = load <2 x double>, <2 x double>* %p, align 32
  ret <2 x double> %v
}

; CHECK-LABEL: store_v2f64_a1:
; CHECK-NEXT: .functype store_v2f64_a1 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0):p2align=0, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v2f64_a1(<2 x double> *%p, <2 x double> %v) {
  store <2 x double> %v, <2 x double>* %p, align 1
  ret void
}

; CHECK-LABEL: store_v2f64_a4:
; CHECK-NEXT: .functype store_v2f64_a4 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0):p2align=2, $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v2f64_a4(<2 x double> *%p, <2 x double> %v) {
  store <2 x double> %v, <2 x double>* %p, align 4
  ret void
}

; 16 is the default alignment for v128 so no attribute is needed.

; CHECK-LABEL: store_v2f64_a16:
; CHECK-NEXT: .functype store_v2f64_a16 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v2f64_a16(<2 x double> *%p, <2 x double> %v) {
  store <2 x double> %v, <2 x double>* %p, align 16
  ret void
}

; 32 is greater than the default alignment so it is ignored.

; CHECK-LABEL: store_v2f64_a32:
; CHECK-NEXT: .functype store_v2f64_a32 (i32, v128) -> (){{$}}
; CHECK-NEXT: v128.store 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @store_v2f64_a32(<2 x double> *%p, <2 x double> %v) {
  store <2 x double> %v, <2 x double>* %p, align 32
  ret void
}
