; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+simd128 | FileCheck %s

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

; 1 is the default alignment for v128.load8_splat so no attribute is needed.

; CHECK-LABEL: load_splat_v16i8_a1:
; CHECK-NEXT: .functype load_splat_v16i8_a1 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load8_splat $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_splat_v16i8_a1(i8* %p) {
  %e = load i8, i8* %p, align 1
  %v1 = insertelement <16 x i8> undef, i8 %e, i32 0
  %v2 = shufflevector <16 x i8> %v1, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %v2
}

; 2 is greater than the default alignment so it is ignored.

; CHECK-LABEL: load_splat_v16i8_a2:
; CHECK-NEXT: .functype load_splat_v16i8_a2 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load8_splat $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_splat_v16i8_a2(i8* %p) {
  %e = load i8, i8* %p, align 2
  %v1 = insertelement <16 x i8> undef, i8 %e, i32 0
  %v2 = shufflevector <16 x i8> %v1, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %v2
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

; CHECK-LABEL: load_ext_v8i16_a1:
; CHECK-NEXT: .functype load_ext_v8i16_a1 (i32) -> (v128){{$}}
; CHECK-NEXT: i16x8.load8x8_u $push[[R:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i8> @load_ext_v8i16_a1(<8 x i8>* %p) {
  %v = load <8 x i8>, <8 x i8>* %p, align 1
  ret <8 x i8> %v
}

; CHECK-LABEL: load_ext_v8i16_a2:
; CHECK-NEXT: .functype load_ext_v8i16_a2 (i32) -> (v128){{$}}
; CHECK-NEXT: i16x8.load8x8_u $push[[R:[0-9]+]]=, 0($0):p2align=1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i8> @load_ext_v8i16_a2(<8 x i8>* %p) {
  %v = load <8 x i8>, <8 x i8>* %p, align 2
  ret <8 x i8> %v
}

; CHECK-LABEL: load_ext_v8i16_a4:
; CHECK-NEXT: .functype load_ext_v8i16_a4 (i32) -> (v128){{$}}
; CHECK-NEXT: i16x8.load8x8_u $push[[R:[0-9]+]]=, 0($0):p2align=2{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i8> @load_ext_v8i16_a4(<8 x i8>* %p) {
  %v = load <8 x i8>, <8 x i8>* %p, align 4
  ret <8 x i8> %v
}

; 8 is the default alignment for v128 extending load so no attribute is needed.

; CHECK-LABEL: load_ext_v8i16_a8:
; CHECK-NEXT: .functype load_ext_v8i16_a8 (i32) -> (v128){{$}}
; CHECK-NEXT: i16x8.load8x8_u $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i8> @load_ext_v8i16_a8(<8 x i8>* %p) {
  %v = load <8 x i8>, <8 x i8>* %p, align 8
  ret <8 x i8> %v
}

; 16 is greater than the default alignment so it is ignored.

; CHECK-LABEL: load_ext_v8i16_a16:
; CHECK-NEXT: .functype load_ext_v8i16_a16 (i32) -> (v128){{$}}
; CHECK-NEXT: i16x8.load8x8_u $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i8> @load_ext_v8i16_a16(<8 x i8>* %p) {
  %v = load <8 x i8>, <8 x i8>* %p, align 16
  ret <8 x i8> %v
}

; CHECK-LABEL: load_sext_v8i16_a1:
; CHECK-NEXT: .functype load_sext_v8i16_a1 (i32) -> (v128){{$}}
; CHECK-NEXT: i16x8.load8x8_s $push[[R:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_sext_v8i16_a1(<8 x i8>* %p) {
  %v = load <8 x i8>, <8 x i8>* %p, align 1
  %v2 = sext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_sext_v8i16_a2:
; CHECK-NEXT: .functype load_sext_v8i16_a2 (i32) -> (v128){{$}}
; CHECK-NEXT: i16x8.load8x8_s $push[[R:[0-9]+]]=, 0($0):p2align=1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_sext_v8i16_a2(<8 x i8>* %p) {
  %v = load <8 x i8>, <8 x i8>* %p, align 2
  %v2 = sext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_sext_v8i16_a4:
; CHECK-NEXT: .functype load_sext_v8i16_a4 (i32) -> (v128){{$}}
; CHECK-NEXT: i16x8.load8x8_s $push[[R:[0-9]+]]=, 0($0):p2align=2{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_sext_v8i16_a4(<8 x i8>* %p) {
  %v = load <8 x i8>, <8 x i8>* %p, align 4
  %v2 = sext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; 8 is the default alignment for v128 extending load so no attribute is needed.

; CHECK-LABEL: load_sext_v8i16_a8:
; CHECK-NEXT: .functype load_sext_v8i16_a8 (i32) -> (v128){{$}}
; CHECK-NEXT: i16x8.load8x8_s $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_sext_v8i16_a8(<8 x i8>* %p) {
  %v = load <8 x i8>, <8 x i8>* %p, align 8
  %v2 = sext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; 16 is greater than the default alignment so it is ignored.

; CHECK-LABEL: load_sext_v8i16_a16:
; CHECK-NEXT: .functype load_sext_v8i16_a16 (i32) -> (v128){{$}}
; CHECK-NEXT: i16x8.load8x8_s $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_sext_v8i16_a16(<8 x i8>* %p) {
  %v = load <8 x i8>, <8 x i8>* %p, align 16
  %v2 = sext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_splat_v8i16_a1:
; CHECK-NEXT: .functype load_splat_v8i16_a1 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load16_splat $push[[R:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_splat_v8i16_a1(i16* %p) {
  %e = load i16, i16* %p, align 1
  %v1 = insertelement <8 x i16> undef, i16 %e, i32 0
  %v2 = shufflevector <8 x i16> %v1, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %v2
}

; 2 is the default alignment for v128.load16_splat so no attribute is needed.

; CHECK-LABEL: load_splat_v8i16_a2:
; CHECK-NEXT: .functype load_splat_v8i16_a2 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load16_splat $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_splat_v8i16_a2(i16* %p) {
  %e = load i16, i16* %p, align 2
  %v1 = insertelement <8 x i16> undef, i16 %e, i32 0
  %v2 = shufflevector <8 x i16> %v1, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %v2
}

; 4 is greater than the default alignment so it is ignored.

; CHECK-LABEL: load_splat_v8i16_a4:
; CHECK-NEXT: .functype load_splat_v8i16_a4 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load16_splat $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_splat_v8i16_a4(i16* %p) {
  %e = load i16, i16* %p, align 4
  %v1 = insertelement <8 x i16> undef, i16 %e, i32 0
  %v2 = shufflevector <8 x i16> %v1, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %v2
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

; CHECK-LABEL: load_ext_v4i32_a1:
; CHECK-NEXT: .functype load_ext_v4i32_a1 (i32) -> (v128){{$}}
; CHECK-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i16> @load_ext_v4i32_a1(<4 x i16>* %p) {
  %v = load <4 x i16>, <4 x i16>* %p, align 1
  ret <4 x i16> %v
}

; CHECK-LABEL: load_ext_v4i32_a2:
; CHECK-NEXT: .functype load_ext_v4i32_a2 (i32) -> (v128){{$}}
; CHECK-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, 0($0):p2align=1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i16> @load_ext_v4i32_a2(<4 x i16>* %p) {
  %v = load <4 x i16>, <4 x i16>* %p, align 2
  ret <4 x i16> %v
}

; CHECK-LABEL: load_ext_v4i32_a4:
; CHECK-NEXT: .functype load_ext_v4i32_a4 (i32) -> (v128){{$}}
; CHECK-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, 0($0):p2align=2{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i16> @load_ext_v4i32_a4(<4 x i16>* %p) {
  %v = load <4 x i16>, <4 x i16>* %p, align 4
  ret <4 x i16> %v
}

; 8 is the default alignment for v128 extending load so no attribute is needed.

; CHECK-LABEL: load_ext_v4i32_a8:
; CHECK-NEXT: .functype load_ext_v4i32_a8 (i32) -> (v128){{$}}
; CHECK-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i16> @load_ext_v4i32_a8(<4 x i16>* %p) {
  %v = load <4 x i16>, <4 x i16>* %p, align 8
  ret <4 x i16> %v
}

; 16 is greater than the default alignment so it is ignored.

; CHECK-LABEL: load_ext_v4i32_a16:
; CHECK-NEXT: .functype load_ext_v4i32_a16 (i32) -> (v128){{$}}
; CHECK-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i16> @load_ext_v4i32_a16(<4 x i16>* %p) {
  %v = load <4 x i16>, <4 x i16>* %p, align 16
  ret <4 x i16> %v
}

; CHECK-LABEL: load_sext_v4i32_a1:
; CHECK-NEXT: .functype load_sext_v4i32_a1 (i32) -> (v128){{$}}
; CHECK-NEXT: i32x4.load16x4_s $push[[R:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_sext_v4i32_a1(<4 x i16>* %p) {
  %v = load <4 x i16>, <4 x i16>* %p, align 1
  %v2 = sext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_sext_v4i32_a2:
; CHECK-NEXT: .functype load_sext_v4i32_a2 (i32) -> (v128){{$}}
; CHECK-NEXT: i32x4.load16x4_s $push[[R:[0-9]+]]=, 0($0):p2align=1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_sext_v4i32_a2(<4 x i16>* %p) {
  %v = load <4 x i16>, <4 x i16>* %p, align 2
  %v2 = sext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_sext_v4i32_a4:
; CHECK-NEXT: .functype load_sext_v4i32_a4 (i32) -> (v128){{$}}
; CHECK-NEXT: i32x4.load16x4_s $push[[R:[0-9]+]]=, 0($0):p2align=2{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_sext_v4i32_a4(<4 x i16>* %p) {
  %v = load <4 x i16>, <4 x i16>* %p, align 4
  %v2 = sext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; 8 is the default alignment for v128 extending load so no attribute is needed.

; CHECK-LABEL: load_sext_v4i32_a8:
; CHECK-NEXT: .functype load_sext_v4i32_a8 (i32) -> (v128){{$}}
; CHECK-NEXT: i32x4.load16x4_s $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_sext_v4i32_a8(<4 x i16>* %p) {
  %v = load <4 x i16>, <4 x i16>* %p, align 8
  %v2 = sext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; 16 is greater than the default alignment so it is ignored.

; CHECK-LABEL: load_sext_v4i32_a16:
; CHECK-NEXT: .functype load_sext_v4i32_a16 (i32) -> (v128){{$}}
; CHECK-NEXT: i32x4.load16x4_s $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_sext_v4i32_a16(<4 x i16>* %p) {
  %v = load <4 x i16>, <4 x i16>* %p, align 16
  %v2 = sext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_splat_v4i32_a1:
; CHECK-NEXT: .functype load_splat_v4i32_a1 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load32_splat $push[[R:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_splat_v4i32_a1(i32* %addr) {
  %e = load i32, i32* %addr, align 1
  %v1 = insertelement <4 x i32> undef, i32 %e, i32 0
  %v2 = shufflevector <4 x i32> %v1, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_splat_v4i32_a2:
; CHECK-NEXT: .functype load_splat_v4i32_a2 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load32_splat $push[[R:[0-9]+]]=, 0($0):p2align=1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_splat_v4i32_a2(i32* %addr) {
  %e = load i32, i32* %addr, align 2
  %v1 = insertelement <4 x i32> undef, i32 %e, i32 0
  %v2 = shufflevector <4 x i32> %v1, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %v2
}

; 4 is the default alignment for v128.load32_splat so no attribute is needed.

; CHECK-LABEL: load_splat_v4i32_a4:
; CHECK-NEXT: .functype load_splat_v4i32_a4 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load32_splat $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_splat_v4i32_a4(i32* %addr) {
  %e = load i32, i32* %addr, align 4
  %v1 = insertelement <4 x i32> undef, i32 %e, i32 0
  %v2 = shufflevector <4 x i32> %v1, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %v2
}

; 8 is greater than the default alignment so it is ignored.

; CHECK-LABEL: load_splat_v4i32_a8:
; CHECK-NEXT: .functype load_splat_v4i32_a8 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load32_splat $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_splat_v4i32_a8(i32* %addr) {
  %e = load i32, i32* %addr, align 8
  %v1 = insertelement <4 x i32> undef, i32 %e, i32 0
  %v2 = shufflevector <4 x i32> %v1, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %v2
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

; CHECK-LABEL: load_splat_v2i64_a1:
; CHECK-NEXT: .functype load_splat_v2i64_a1 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load64_splat $push[[R:[0-9]+]]=, 0($0):p2align=0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_splat_v2i64_a1(i64* %p) {
  %e = load i64, i64* %p, align 1
  %v1 = insertelement <2 x i64> undef, i64 %e, i32 0
  %v2 = shufflevector <2 x i64> %v1, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_splat_v2i64_a2:
; CHECK-NEXT: .functype load_splat_v2i64_a2 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load64_splat $push[[R:[0-9]+]]=, 0($0):p2align=1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_splat_v2i64_a2(i64* %p) {
  %e = load i64, i64* %p, align 2
  %v1 = insertelement <2 x i64> undef, i64 %e, i32 0
  %v2 = shufflevector <2 x i64> %v1, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_splat_v2i64_a4:
; CHECK-NEXT: .functype load_splat_v2i64_a4 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load64_splat $push[[R:[0-9]+]]=, 0($0):p2align=2{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_splat_v2i64_a4(i64* %p) {
  %e = load i64, i64* %p, align 4
  %v1 = insertelement <2 x i64> undef, i64 %e, i32 0
  %v2 = shufflevector <2 x i64> %v1, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %v2
}

; 8 is the default alignment for v128.load64_splat so no attribute is needed.

; CHECK-LABEL: load_splat_v2i64_a8:
; CHECK-NEXT: .functype load_splat_v2i64_a8 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load64_splat $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_splat_v2i64_a8(i64* %p) {
  %e = load i64, i64* %p, align 8
  %v1 = insertelement <2 x i64> undef, i64 %e, i32 0
  %v2 = shufflevector <2 x i64> %v1, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %v2
}

; 16 is greater than the default alignment so it is ignored.

; CHECK-LABEL: load_splat_v2i64_a16:
; CHECK-NEXT: .functype load_splat_v2i64_a16 (i32) -> (v128){{$}}
; CHECK-NEXT: v128.load64_splat $push[[R:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_splat_v2i64_a16(i64* %p) {
  %e = load i64, i64* %p, align 16
  %v1 = insertelement <2 x i64> undef, i64 %e, i32 0
  %v2 = shufflevector <2 x i64> %v1, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %v2
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
