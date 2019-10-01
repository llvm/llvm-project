; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-keep-registers -wasm-disable-explicit-locals -mattr=+unimplemented-simd128 | FileCheck %s --check-prefixes CHECK,SIMD128
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-keep-registers -wasm-disable-explicit-locals -mattr=+simd128 | FileCheck %s --check-prefixes CHECK,SIMD128-VM
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-keep-registers -wasm-disable-explicit-locals | FileCheck %s --check-prefixes CHECK,NO-SIMD128

; Test SIMD loads and stores

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; ==============================================================================
; 16 x i8
; ==============================================================================
; CHECK-LABEL: load_v16i8:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v16i8 (i32) -> (v128){{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_v16i8(<16 x i8>* %p) {
  %v = load <16 x i8>, <16 x i8>* %p
  ret <16 x i8> %v
}

; CHECK-LABEL: load_splat_v16i8:
; SIMD128-VM-NOT: v8x16.load_splat
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v16i8 (i32) -> (v128){{$}}
; SIMD128-NEXT: v8x16.load_splat $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_splat_v16i8(i8* %p) {
  %e = load i8, i8* %p
  %v1 = insertelement <16 x i8> undef, i8 %e, i32 0
  %v2 = shufflevector <16 x i8> %v1, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %v2
}

; CHECK-LABEL: load_v16i8_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v16i8_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_v16i8_with_folded_offset(<16 x i8>* %p) {
  %q = ptrtoint <16 x i8>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <16 x i8>*
  %v = load <16 x i8>, <16 x i8>* %s
  ret <16 x i8> %v
}

; CHECK-LABEL: load_splat_v16i8_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v16i8_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v8x16.load_splat $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_splat_v16i8_with_folded_offset(i8* %p) {
  %q = ptrtoint i8* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to i8*
  %e = load i8, i8* %s
  %v1 = insertelement <16 x i8> undef, i8 %e, i32 0
  %v2 = shufflevector <16 x i8> %v1, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %v2
}

; CHECK-LABEL: load_v16i8_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v16i8_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_v16i8_with_folded_gep_offset(<16 x i8>* %p) {
  %s = getelementptr inbounds <16 x i8>, <16 x i8>* %p, i32 1
  %v = load <16 x i8>, <16 x i8>* %s
  ret <16 x i8> %v
}

; CHECK-LABEL: load_splat_v16i8_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v16i8_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v8x16.load_splat $push[[R:[0-9]+]]=, 1($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_splat_v16i8_with_folded_gep_offset(i8* %p) {
  %s = getelementptr inbounds i8, i8* %p, i32 1
  %e = load i8, i8* %s
  %v1 = insertelement <16 x i8> undef, i8 %e, i32 0
  %v2 = shufflevector <16 x i8> %v1, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %v2
}

; CHECK-LABEL: load_v16i8_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v16i8_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_v16i8_with_unfolded_gep_negative_offset(<16 x i8>* %p) {
  %s = getelementptr inbounds <16 x i8>, <16 x i8>* %p, i32 -1
  %v = load <16 x i8>, <16 x i8>* %s
  ret <16 x i8> %v
}

; CHECK-LABEL: load_splat_v16i8_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v16i8_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -1{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v8x16.load_splat $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_splat_v16i8_with_unfolded_gep_negative_offset(i8* %p) {
  %s = getelementptr inbounds i8, i8* %p, i32 -1
  %e = load i8, i8* %s
  %v1 = insertelement <16 x i8> undef, i8 %e, i32 0
  %v2 = shufflevector <16 x i8> %v1, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %v2
}

; CHECK-LABEL: load_v16i8_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v16i8_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_v16i8_with_unfolded_offset(<16 x i8>* %p) {
  %q = ptrtoint <16 x i8>* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to <16 x i8>*
  %v = load <16 x i8>, <16 x i8>* %s
  ret <16 x i8> %v
}

; CHECK-LABEL: load_splat_v16i8_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v16i8_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v8x16.load_splat $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_splat_v16i8_with_unfolded_offset(i8* %p) {
  %q = ptrtoint i8* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to i8*
  %e = load i8, i8* %s
  %v1 = insertelement <16 x i8> undef, i8 %e, i32 0
  %v2 = shufflevector <16 x i8> %v1, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %v2
}

; CHECK-LABEL: load_v16i8_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v16i8_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_v16i8_with_unfolded_gep_offset(<16 x i8>* %p) {
  %s = getelementptr <16 x i8>, <16 x i8>* %p, i32 1
  %v = load <16 x i8>, <16 x i8>* %s
  ret <16 x i8> %v
}

; CHECK-LABEL: load_splat_v16i8_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v16i8_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 1{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v8x16.load_splat $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_splat_v16i8_with_unfolded_gep_offset(i8* %p) {
  %s = getelementptr i8, i8* %p, i32 1
  %e = load i8, i8* %s
  %v1 = insertelement <16 x i8> undef, i8 %e, i32 0
  %v2 = shufflevector <16 x i8> %v1, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %v2
}

; CHECK-LABEL: load_v16i8_from_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v16i8_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_v16i8_from_numeric_address() {
  %s = inttoptr i32 32 to <16 x i8>*
  %v = load <16 x i8>, <16 x i8>* %s
  ret <16 x i8> %v
}

; CHECK-LABEL: load_splat_v16i8_from_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v16i8_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v8x16.load_splat $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @load_splat_v16i8_from_numeric_address() {
  %s = inttoptr i32 32 to i8*
  %e = load i8, i8* %s
  %v1 = insertelement <16 x i8> undef, i8 %e, i32 0
  %v2 = shufflevector <16 x i8> %v1, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %v2
}

; CHECK-LABEL: load_v16i8_from_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v16i8_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, gv_v16i8($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
@gv_v16i8 = global <16 x i8> <i8 42, i8 42, i8 42, i8 42, i8 42, i8 42, i8 42, i8 42, i8 42, i8 42, i8 42, i8 42, i8 42, i8 42, i8 42, i8 42>
define <16 x i8> @load_v16i8_from_global_address() {
  %v = load <16 x i8>, <16 x i8>* @gv_v16i8
  ret <16 x i8> %v
}

; CHECK-LABEL: load_splat_v16i8_from_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v16i8_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v8x16.load_splat $push[[R:[0-9]+]]=, gv_i8($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
@gv_i8 = global i8 42
define <16 x i8> @load_splat_v16i8_from_global_address() {
  %e = load i8, i8* @gv_i8
  %v1 = insertelement <16 x i8> undef, i8 %e, i32 0
  %v2 = shufflevector <16 x i8> %v1, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %v2
}

; CHECK-LABEL: store_v16i8:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v16i8 (v128, i32) -> (){{$}}
; SIMD128-NEXT: v128.store 0($1), $0{{$}}
define void @store_v16i8(<16 x i8> %v, <16 x i8>* %p) {
  store <16 x i8> %v , <16 x i8>* %p
  ret void
}

; CHECK-LABEL: store_v16i8_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v16i8_with_folded_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: v128.store 16($1), $0{{$}}
define void @store_v16i8_with_folded_offset(<16 x i8> %v, <16 x i8>* %p) {
  %q = ptrtoint <16 x i8>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <16 x i8>*
  store <16 x i8> %v , <16 x i8>* %s
  ret void
}

; CHECK-LABEL: store_v16i8_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v16i8_with_folded_gep_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: v128.store 16($1), $0{{$}}
define void @store_v16i8_with_folded_gep_offset(<16 x i8> %v, <16 x i8>* %p) {
  %s = getelementptr inbounds <16 x i8>, <16 x i8>* %p, i32 1
  store <16 x i8> %v , <16 x i8>* %s
  ret void
}

; CHECK-LABEL: store_v16i8_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v16i8_with_unfolded_gep_negative_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -16{{$}}
; SIMD128-NEXT: i32.add $push[[R:[0-9]+]]=, $1, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[R]]), $0{{$}}
define void @store_v16i8_with_unfolded_gep_negative_offset(<16 x i8> %v, <16 x i8>* %p) {
  %s = getelementptr inbounds <16 x i8>, <16 x i8>* %p, i32 -1
  store <16 x i8> %v , <16 x i8>* %s
  ret void
}

; CHECK-LABEL: store_v16i8_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v16i8_with_unfolded_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -16{{$}}
; SIMD128-NEXT: i32.add $push[[R:[0-9]+]]=, $1, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[R]]), $0{{$}}
define void @store_v16i8_with_unfolded_offset(<16 x i8> %v, <16 x i8>* %p) {
  %s = getelementptr inbounds <16 x i8>, <16 x i8>* %p, i32 -1
  store <16 x i8> %v , <16 x i8>* %s
  ret void
}

; CHECK-LABEL: store_v16i8_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v16i8_with_unfolded_gep_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[R:[0-9]+]]=, $1, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[R]]), $0{{$}}
define void @store_v16i8_with_unfolded_gep_offset(<16 x i8> %v, <16 x i8>* %p) {
  %s = getelementptr <16 x i8>, <16 x i8>* %p, i32 1
  store <16 x i8> %v , <16 x i8>* %s
  ret void
}

; CHECK-LABEL: store_v16i8_to_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v16i8_to_numeric_address (v128) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[R:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.store 32($pop[[R]]), $0{{$}}
define void @store_v16i8_to_numeric_address(<16 x i8> %v) {
  %s = inttoptr i32 32 to <16 x i8>*
  store <16 x i8> %v , <16 x i8>* %s
  ret void
}

; CHECK-LABEL: store_v16i8_to_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v16i8_to_global_address (v128) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[R:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.store gv_v16i8($pop[[R]]), $0{{$}}
define void @store_v16i8_to_global_address(<16 x i8> %v) {
  store <16 x i8> %v , <16 x i8>* @gv_v16i8
  ret void
}

; ==============================================================================
; 8 x i16
; ==============================================================================
; CHECK-LABEL: load_v8i16:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v8i16 (i32) -> (v128){{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_v8i16(<8 x i16>* %p) {
  %v = load <8 x i16>, <8 x i16>* %p
  ret <8 x i16> %v
}

; CHECK-LABEL: load_splat_v8i16:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v8i16 (i32) -> (v128){{$}}
; SIMD128-NEXT: v16x8.load_splat $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_splat_v8i16(i16* %p) {
  %e = load i16, i16* %p
  %v1 = insertelement <8 x i16> undef, i16 %e, i32 0
  %v2 = shufflevector <8 x i16> %v1, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_sext_v8i16:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_sext_v8i16 (i32) -> (v128){{$}}
; SIMD128-NEXT: i16x8.load8x8_s $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_sext_v8i16(<8 x i8>* %p) {
  %v = load <8 x i8>, <8 x i8>* %p
  %v2 = sext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_zext_v8i16:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_zext_v8i16 (i32) -> (v128){{$}}
; SIMD128-NEXT: i16x8.load8x8_u $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_zext_v8i16(<8 x i8>* %p) {
  %v = load <8 x i8>, <8 x i8>* %p
  %v2 = zext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_ext_v8i16:
; NO-SIMD128-NOT: load8x8
; SIMD128-NEXT: .functype load_ext_v8i16 (i32) -> (v128){{$}}
; SIMD128-NEXT: i16x8.load8x8_u $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i8> @load_ext_v8i16(<8 x i8>* %p) {
  %v = load <8 x i8>, <8 x i8>* %p
  ret <8 x i8> %v
}

; CHECK-LABEL: load_v8i16_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v8i16_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_v8i16_with_folded_offset(<8 x i16>* %p) {
  %q = ptrtoint <8 x i16>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <8 x i16>*
  %v = load <8 x i16>, <8 x i16>* %s
  ret <8 x i16> %v
}

; CHECK-LABEL: load_splat_v8i16_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v8i16_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v16x8.load_splat $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_splat_v8i16_with_folded_offset(i16* %p) {
  %q = ptrtoint i16* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to i16*
  %e = load i16, i16* %s
  %v1 = insertelement <8 x i16> undef, i16 %e, i32 0
  %v2 = shufflevector <8 x i16> %v1, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_sext_v8i16_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_sext_v8i16_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i16x8.load8x8_s $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_sext_v8i16_with_folded_offset(<8 x i8>* %p) {
  %q = ptrtoint <8 x i8>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <8 x i8>*
  %v = load <8 x i8>, <8 x i8>* %s
  %v2 = sext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_zext_v8i16_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_zext_v8i16_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i16x8.load8x8_u $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_zext_v8i16_with_folded_offset(<8 x i8>* %p) {
  %q = ptrtoint <8 x i8>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <8 x i8>*
  %v = load <8 x i8>, <8 x i8>* %s
  %v2 = zext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_ext_v8i16_with_folded_offset:
; NO-SIMD128-NOT: load8x8
; SIMD128-NEXT: .functype load_ext_v8i16_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i16x8.load8x8_u $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i8> @load_ext_v8i16_with_folded_offset(<8 x i8>* %p) {
  %q = ptrtoint <8 x i8>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <8 x i8>*
  %v = load <8 x i8>, <8 x i8>* %s
  ret <8 x i8> %v
}

; CHECK-LABEL: load_v8i16_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v8i16_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_v8i16_with_folded_gep_offset(<8 x i16>* %p) {
  %s = getelementptr inbounds <8 x i16>, <8 x i16>* %p, i32 1
  %v = load <8 x i16>, <8 x i16>* %s
  ret <8 x i16> %v
}

; CHECK-LABEL: load_splat_v8i16_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v8i16_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v16x8.load_splat $push[[R:[0-9]+]]=, 2($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_splat_v8i16_with_folded_gep_offset(i16* %p) {
  %s = getelementptr inbounds i16, i16* %p, i32 1
  %e = load i16, i16* %s
  %v1 = insertelement <8 x i16> undef, i16 %e, i32 0
  %v2 = shufflevector <8 x i16> %v1, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_sext_v8i16_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_sext_v8i16_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i16x8.load8x8_s $push[[R:[0-9]+]]=, 8($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_sext_v8i16_with_folded_gep_offset(<8 x i8>* %p) {
  %s = getelementptr inbounds <8 x i8>, <8 x i8>* %p, i32 1
  %v = load <8 x i8>, <8 x i8>* %s
  %v2 = sext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_zext_v8i16_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_zext_v8i16_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i16x8.load8x8_u $push[[R:[0-9]+]]=, 8($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_zext_v8i16_with_folded_gep_offset(<8 x i8>* %p) {
  %s = getelementptr inbounds <8 x i8>, <8 x i8>* %p, i32 1
  %v = load <8 x i8>, <8 x i8>* %s
  %v2 = zext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_ext_v8i16_with_folded_gep_offset:
; NO-SIMD128-NOT: load8x8
; SIMD128-NEXT: .functype load_ext_v8i16_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i16x8.load8x8_u $push[[R:[0-9]+]]=, 8($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i8> @load_ext_v8i16_with_folded_gep_offset(<8 x i8>* %p) {
  %s = getelementptr inbounds <8 x i8>, <8 x i8>* %p, i32 1
  %v = load <8 x i8>, <8 x i8>* %s
  ret <8 x i8> %v
}

; CHECK-LABEL: load_v8i16_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v8i16_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_v8i16_with_unfolded_gep_negative_offset(<8 x i16>* %p) {
  %s = getelementptr inbounds <8 x i16>, <8 x i16>* %p, i32 -1
  %v = load <8 x i16>, <8 x i16>* %s
  ret <8 x i16> %v
}

; CHECK-LABEL: load_splat_v8i16_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v8i16_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -2{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v16x8.load_splat $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_splat_v8i16_with_unfolded_gep_negative_offset(i16* %p) {
  %s = getelementptr inbounds i16, i16* %p, i32 -1
  %e = load i16, i16* %s
  %v1 = insertelement <8 x i16> undef, i16 %e, i32 0
  %v2 = shufflevector <8 x i16> %v1, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_sext_v8i16_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_sext_v8i16_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i16x8.load8x8_s $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_sext_v8i16_with_unfolded_gep_negative_offset(<8 x i8>* %p) {
  %s = getelementptr inbounds <8 x i8>, <8 x i8>* %p, i32 -1
  %v = load <8 x i8>, <8 x i8>* %s
  %v2 = sext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_zext_v8i16_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_zext_v8i16_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i16x8.load8x8_u $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_zext_v8i16_with_unfolded_gep_negative_offset(<8 x i8>* %p) {
  %s = getelementptr inbounds <8 x i8>, <8 x i8>* %p, i32 -1
  %v = load <8 x i8>, <8 x i8>* %s
  %v2 = zext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_ext_v8i16_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: load8x8
; SIMD128-NEXT: .functype load_ext_v8i16_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i16x8.load8x8_u $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i8> @load_ext_v8i16_with_unfolded_gep_negative_offset(<8 x i8>* %p) {
  %s = getelementptr inbounds <8 x i8>, <8 x i8>* %p, i32 -1
  %v = load <8 x i8>, <8 x i8>* %s
  ret <8 x i8> %v
}

; CHECK-LABEL: load_v8i16_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v8i16_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.load $push[[L0:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[L0]]{{$}}
define <8 x i16> @load_v8i16_with_unfolded_offset(<8 x i16>* %p) {
  %q = ptrtoint <8 x i16>* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to <8 x i16>*
  %v = load <8 x i16>, <8 x i16>* %s
  ret <8 x i16> %v
}

; CHECK-LABEL: load_splat_v8i16_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v8i16_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v16x8.load_splat $push[[L0:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[L0]]{{$}}
define <8 x i16> @load_splat_v8i16_with_unfolded_offset(i16* %p) {
  %q = ptrtoint i16* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to i16*
  %e = load i16, i16* %s
  %v1 = insertelement <8 x i16> undef, i16 %e, i32 0
  %v2 = shufflevector <8 x i16> %v1, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_sext_v8i16_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_sext_v8i16_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i16x8.load8x8_s $push[[L0:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[L0]]{{$}}
define <8 x i16> @load_sext_v8i16_with_unfolded_offset(<8 x i8>* %p) {
  %q = ptrtoint <8 x i8>* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to <8 x i8>*
  %v = load <8 x i8>, <8 x i8>* %s
  %v2 = sext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_zext_v8i16_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_zext_v8i16_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i16x8.load8x8_u $push[[L0:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[L0]]{{$}}
define <8 x i16> @load_zext_v8i16_with_unfolded_offset(<8 x i8>* %p) {
  %q = ptrtoint <8 x i8>* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to <8 x i8>*
  %v = load <8 x i8>, <8 x i8>* %s
  %v2 = zext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_ext_v8i16_with_unfolded_offset:
; NO-SIMD128-NOT: load8x8
; SIMD128-NEXT: .functype load_ext_v8i16_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i16x8.load8x8_u $push[[L0:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[L0]]{{$}}
define <8 x i8> @load_ext_v8i16_with_unfolded_offset(<8 x i8>* %p) {
  %q = ptrtoint <8 x i8>* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to <8 x i8>*
  %v = load <8 x i8>, <8 x i8>* %s
  ret <8 x i8> %v
}

; CHECK-LABEL: load_v8i16_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v8i16_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_v8i16_with_unfolded_gep_offset(<8 x i16>* %p) {
  %s = getelementptr <8 x i16>, <8 x i16>* %p, i32 1
  %v = load <8 x i16>, <8 x i16>* %s
  ret <8 x i16> %v
}

; CHECK-LABEL: load_splat_v8i16_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v8i16_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 2{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v16x8.load_splat $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_splat_v8i16_with_unfolded_gep_offset(i16* %p) {
  %s = getelementptr i16, i16* %p, i32 1
  %e = load i16, i16* %s
  %v1 = insertelement <8 x i16> undef, i16 %e, i32 0
  %v2 = shufflevector <8 x i16> %v1, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_sext_v8i16_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_sext_v8i16_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i16x8.load8x8_s $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_sext_v8i16_with_unfolded_gep_offset(<8 x i8>* %p) {
  %s = getelementptr <8 x i8>, <8 x i8>* %p, i32 1
  %v = load <8 x i8>, <8 x i8>* %s
  %v2 = sext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_zext_v8i16_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_zext_v8i16_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i16x8.load8x8_u $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_zext_v8i16_with_unfolded_gep_offset(<8 x i8>* %p) {
  %s = getelementptr <8 x i8>, <8 x i8>* %p, i32 1
  %v = load <8 x i8>, <8 x i8>* %s
  %v2 = zext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_ext_v8i16_with_unfolded_gep_offset:
; NO-SIMD128-NOT: load8x8
; SIMD128-NEXT: .functype load_ext_v8i16_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i16x8.load8x8_u $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i8> @load_ext_v8i16_with_unfolded_gep_offset(<8 x i8>* %p) {
  %s = getelementptr <8 x i8>, <8 x i8>* %p, i32 1
  %v = load <8 x i8>, <8 x i8>* %s
  ret <8 x i8> %v
}

; CHECK-LABEL: load_v8i16_from_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v8i16_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_v8i16_from_numeric_address() {
  %s = inttoptr i32 32 to <8 x i16>*
  %v = load <8 x i16>, <8 x i16>* %s
  ret <8 x i16> %v
}

; CHECK-LABEL: load_splat_v8i16_from_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v8i16_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v16x8.load_splat $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_splat_v8i16_from_numeric_address() {
  %s = inttoptr i32 32 to i16*
  %e = load i16, i16* %s
  %v1 = insertelement <8 x i16> undef, i16 %e, i32 0
  %v2 = shufflevector <8 x i16> %v1, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_sext_v8i16_from_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_sext_v8i16_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: i16x8.load8x8_s $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_sext_v8i16_from_numeric_address() {
  %s = inttoptr i32 32 to <8 x i8>*
  %v = load <8 x i8>, <8 x i8>* %s
  %v2 = sext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_zext_v8i16_from_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_zext_v8i16_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: i16x8.load8x8_u $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_zext_v8i16_from_numeric_address() {
  %s = inttoptr i32 32 to <8 x i8>*
  %v = load <8 x i8>, <8 x i8>* %s
  %v2 = zext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_ext_v8i16_from_numeric_address:
; NO-SIMD128-NOT: load8x8
; SIMD128-NEXT: .functype load_ext_v8i16_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: i16x8.load8x8_u $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i8> @load_ext_v8i16_from_numeric_address() {
  %s = inttoptr i32 32 to <8 x i8>*
  %v = load <8 x i8>, <8 x i8>* %s
  ret <8 x i8> %v
}

; CHECK-LABEL: load_v8i16_from_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v8i16_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, gv_v8i16($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
@gv_v8i16 = global <8 x i16> <i16 42, i16 42, i16 42, i16 42, i16 42, i16 42, i16 42, i16 42>
define <8 x i16> @load_v8i16_from_global_address() {
  %v = load <8 x i16>, <8 x i16>* @gv_v8i16
  ret <8 x i16> %v
}

; CHECK-LABEL: load_splat_v8i16_from_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v8i16_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v16x8.load_splat $push[[R:[0-9]+]]=, gv_i16($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
@gv_i16 = global i16 42
define <8 x i16> @load_splat_v8i16_from_global_address() {
  %e = load i16, i16* @gv_i16
  %v1 = insertelement <8 x i16> undef, i16 %e, i32 0
  %v2 = shufflevector <8 x i16> %v1, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_sext_v8i16_from_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_sext_v8i16_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: i16x8.load8x8_s $push[[R:[0-9]+]]=, gv_v8i8($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
@gv_v8i8 = global <8 x i8> <i8 42, i8 42, i8 42, i8 42, i8 42, i8 42, i8 42, i8 42>
define <8 x i16> @load_sext_v8i16_from_global_address() {
  %v = load <8 x i8>, <8 x i8>* @gv_v8i8
  %v2 = sext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_zext_v8i16_from_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_zext_v8i16_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: i16x8.load8x8_u $push[[R:[0-9]+]]=, gv_v8i8($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @load_zext_v8i16_from_global_address() {
  %v = load <8 x i8>, <8 x i8>* @gv_v8i8
  %v2 = zext <8 x i8> %v to <8 x i16>
  ret <8 x i16> %v2
}

; CHECK-LABEL: load_ext_v8i16_from_global_address:
; NO-SIMD128-NOT: load8x8
; SIMD128-NEXT: .functype load_ext_v8i16_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: i16x8.load8x8_u $push[[R:[0-9]+]]=, gv_v8i8($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i8> @load_ext_v8i16_from_global_address() {
  %v = load <8 x i8>, <8 x i8>* @gv_v8i8
  ret <8 x i8> %v
}


; CHECK-LABEL: store_v8i16:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v8i16 (v128, i32) -> (){{$}}
; SIMD128-NEXT: v128.store 0($1), $0{{$}}
define void @store_v8i16(<8 x i16> %v, <8 x i16>* %p) {
  store <8 x i16> %v , <8 x i16>* %p
  ret void
}

; CHECK-LABEL: store_v8i16_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v8i16_with_folded_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: v128.store 16($1), $0{{$}}
define void @store_v8i16_with_folded_offset(<8 x i16> %v, <8 x i16>* %p) {
  %q = ptrtoint <8 x i16>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <8 x i16>*
  store <8 x i16> %v , <8 x i16>* %s
  ret void
}

; CHECK-LABEL: store_v8i16_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v8i16_with_folded_gep_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: v128.store 16($1), $0{{$}}
define void @store_v8i16_with_folded_gep_offset(<8 x i16> %v, <8 x i16>* %p) {
  %s = getelementptr inbounds <8 x i16>, <8 x i16>* %p, i32 1
  store <8 x i16> %v , <8 x i16>* %s
  ret void
}

; CHECK-LABEL: store_v8i16_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v8i16_with_unfolded_gep_negative_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -16{{$}}
; SIMD128-NEXT: i32.add $push[[R:[0-9]+]]=, $1, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[R]]), $0{{$}}
define void @store_v8i16_with_unfolded_gep_negative_offset(<8 x i16> %v, <8 x i16>* %p) {
  %s = getelementptr inbounds <8 x i16>, <8 x i16>* %p, i32 -1
  store <8 x i16> %v , <8 x i16>* %s
  ret void
}

; CHECK-LABEL: store_v8i16_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v8i16_with_unfolded_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -16{{$}}
; SIMD128-NEXT: i32.add $push[[R:[0-9]+]]=, $1, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[R]]), $0{{$}}
define void @store_v8i16_with_unfolded_offset(<8 x i16> %v, <8 x i16>* %p) {
  %s = getelementptr inbounds <8 x i16>, <8 x i16>* %p, i32 -1
  store <8 x i16> %v , <8 x i16>* %s
  ret void
}

; CHECK-LABEL: store_v8i16_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v8i16_with_unfolded_gep_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[R:[0-9]+]]=, $1, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[R]]), $0{{$}}
define void @store_v8i16_with_unfolded_gep_offset(<8 x i16> %v, <8 x i16>* %p) {
  %s = getelementptr <8 x i16>, <8 x i16>* %p, i32 1
  store <8 x i16> %v , <8 x i16>* %s
  ret void
}

; CHECK-LABEL: store_v8i16_to_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v8i16_to_numeric_address (v128) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.store 32($pop[[L0]]), $0{{$}}
define void @store_v8i16_to_numeric_address(<8 x i16> %v) {
  %s = inttoptr i32 32 to <8 x i16>*
  store <8 x i16> %v , <8 x i16>* %s
  ret void
}

; CHECK-LABEL: store_v8i16_to_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v8i16_to_global_address (v128) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[R:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.store gv_v8i16($pop[[R]]), $0{{$}}
define void @store_v8i16_to_global_address(<8 x i16> %v) {
  store <8 x i16> %v , <8 x i16>* @gv_v8i16
  ret void
}

; ==============================================================================
; 4 x i32
; ==============================================================================
; CHECK-LABEL: load_v4i32:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v4i32 (i32) -> (v128){{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_v4i32(<4 x i32>* %p) {
  %v = load <4 x i32>, <4 x i32>* %p
  ret <4 x i32> %v
}

; CHECK-LABEL: load_splat_v4i32:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v4i32 (i32) -> (v128){{$}}
; SIMD128-NEXT: v32x4.load_splat
define <4 x i32> @load_splat_v4i32(i32* %addr) {
  %e = load i32, i32* %addr, align 4
  %v1 = insertelement <4 x i32> undef, i32 %e, i32 0
  %v2 = shufflevector <4 x i32> %v1, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_sext_v4i32:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_sext_v4i32 (i32) -> (v128){{$}}
; SIMD128-NEXT: i32x4.load16x4_s $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_sext_v4i32(<4 x i16>* %p) {
  %v = load <4 x i16>, <4 x i16>* %p
  %v2 = sext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_zext_v4i32:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_zext_v4i32 (i32) -> (v128){{$}}
; SIMD128-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_zext_v4i32(<4 x i16>* %p) {
  %v = load <4 x i16>, <4 x i16>* %p
  %v2 = zext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_ext_v4i32:
; NO-SIMD128-NOT: load16x4
; SIMD128-NEXT: .functype load_ext_v4i32 (i32) -> (v128){{$}}
; SIMD128-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i16> @load_ext_v4i32(<4 x i16>* %p) {
  %v = load <4 x i16>, <4 x i16>* %p
  ret <4 x i16> %v
}

; CHECK-LABEL: load_v4i32_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v4i32_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_v4i32_with_folded_offset(<4 x i32>* %p) {
  %q = ptrtoint <4 x i32>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <4 x i32>*
  %v = load <4 x i32>, <4 x i32>* %s
  ret <4 x i32> %v
}

; CHECK-LABEL: load_splat_v4i32_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v4i32_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v32x4.load_splat $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_splat_v4i32_with_folded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to i32*
  %e = load i32, i32* %s
  %v1 = insertelement <4 x i32> undef, i32 %e, i32 0
  %v2 = shufflevector <4 x i32> %v1, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_sext_v4i32_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_sext_v4i32_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32x4.load16x4_s $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_sext_v4i32_with_folded_offset(<4 x i16>* %p) {
  %q = ptrtoint <4 x i16>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <4 x i16>*
  %v = load <4 x i16>, <4 x i16>* %s
  %v2 = sext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_zext_v4i32_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_zext_v4i32_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_zext_v4i32_with_folded_offset(<4 x i16>* %p) {
  %q = ptrtoint <4 x i16>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <4 x i16>*
  %v = load <4 x i16>, <4 x i16>* %s
  %v2 = zext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_ext_v4i32_with_folded_offset:
; NO-SIMD128-NOT: load16x4
; SIMD128-NEXT: .functype load_ext_v4i32_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i16> @load_ext_v4i32_with_folded_offset(<4 x i16>* %p) {
  %q = ptrtoint <4 x i16>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <4 x i16>*
  %v = load <4 x i16>, <4 x i16>* %s
  ret <4 x i16> %v
}

; CHECK-LABEL: load_v4i32_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v4i32_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_v4i32_with_folded_gep_offset(<4 x i32>* %p) {
  %s = getelementptr inbounds <4 x i32>, <4 x i32>* %p, i32 1
  %v = load <4 x i32>, <4 x i32>* %s
  ret <4 x i32> %v
}

; CHECK-LABEL: load_splat_v4i32_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v4i32_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v32x4.load_splat $push[[R:[0-9]+]]=, 4($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_splat_v4i32_with_folded_gep_offset(i32* %p) {
  %s = getelementptr inbounds i32, i32* %p, i32 1
  %e = load i32, i32* %s
  %v1 = insertelement <4 x i32> undef, i32 %e, i32 0
  %v2 = shufflevector <4 x i32> %v1, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_sext_v4i32_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_sext_v4i32_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32x4.load16x4_s $push[[R:[0-9]+]]=, 8($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_sext_v4i32_with_folded_gep_offset(<4 x i16>* %p) {
  %s = getelementptr inbounds <4 x i16>, <4 x i16>* %p, i32 1
  %v = load <4 x i16>, <4 x i16>* %s
  %v2 = sext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_zext_v4i32_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_zext_v4i32_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, 8($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_zext_v4i32_with_folded_gep_offset(<4 x i16>* %p) {
  %s = getelementptr inbounds <4 x i16>, <4 x i16>* %p, i32 1
  %v = load <4 x i16>, <4 x i16>* %s
  %v2 = zext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_ext_v4i32_with_folded_gep_offset:
; NO-SIMD128-NOT: load16x4
; SIMD128-NEXT: .functype load_ext_v4i32_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, 8($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i16> @load_ext_v4i32_with_folded_gep_offset(<4 x i16>* %p) {
  %s = getelementptr inbounds <4 x i16>, <4 x i16>* %p, i32 1
  %v = load <4 x i16>, <4 x i16>* %s
  ret <4 x i16> %v
}

; CHECK-LABEL: load_v4i32_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v4i32_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_v4i32_with_unfolded_gep_negative_offset(<4 x i32>* %p) {
  %s = getelementptr inbounds <4 x i32>, <4 x i32>* %p, i32 -1
  %v = load <4 x i32>, <4 x i32>* %s
  ret <4 x i32> %v
}

; CHECK-LABEL: load_splat_v4i32_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v4i32_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -4{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v32x4.load_splat $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_splat_v4i32_with_unfolded_gep_negative_offset(i32* %p) {
  %s = getelementptr inbounds i32, i32* %p, i32 -1
  %e = load i32, i32* %s
  %v1 = insertelement <4 x i32> undef, i32 %e, i32 0
  %v2 = shufflevector <4 x i32> %v1, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_sext_v4i32_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_sext_v4i32_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i32x4.load16x4_s $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_sext_v4i32_with_unfolded_gep_negative_offset(<4 x i16>* %p) {
  %s = getelementptr inbounds <4 x i16>, <4 x i16>* %p, i32 -1
  %v = load <4 x i16>, <4 x i16>* %s
  %v2 = sext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_zext_v4i32_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_zext_v4i32_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_zext_v4i32_with_unfolded_gep_negative_offset(<4 x i16>* %p) {
  %s = getelementptr inbounds <4 x i16>, <4 x i16>* %p, i32 -1
  %v = load <4 x i16>, <4 x i16>* %s
  %v2 = zext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_ext_v4i32_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: load16x4
; SIMD128-NEXT: .functype load_ext_v4i32_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i16> @load_ext_v4i32_with_unfolded_gep_negative_offset(<4 x i16>* %p) {
  %s = getelementptr inbounds <4 x i16>, <4 x i16>* %p, i32 -1
  %v = load <4 x i16>, <4 x i16>* %s
  ret <4 x i16> %v
}

; CHECK-LABEL: load_v4i32_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v4i32_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_v4i32_with_unfolded_offset(<4 x i32>* %p) {
  %q = ptrtoint <4 x i32>* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to <4 x i32>*
  %v = load <4 x i32>, <4 x i32>* %s
  ret <4 x i32> %v
}

; CHECK-LABEL: load_splat_v4i32_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v4i32_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v32x4.load_splat $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_splat_v4i32_with_unfolded_offset(i32* %p) {
  %q = ptrtoint i32* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to i32*
  %e = load i32, i32* %s
  %v1 = insertelement <4 x i32> undef, i32 %e, i32 0
  %v2 = shufflevector <4 x i32> %v1, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_sext_v4i32_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_sext_v4i32_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i32x4.load16x4_s $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_sext_v4i32_with_unfolded_offset(<4 x i16>* %p) {
  %q = ptrtoint <4 x i16>* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to <4 x i16>*
  %v = load <4 x i16>, <4 x i16>* %s
  %v2 = sext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_zext_v4i32_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_zext_v4i32_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_zext_v4i32_with_unfolded_offset(<4 x i16>* %p) {
  %q = ptrtoint <4 x i16>* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to <4 x i16>*
  %v = load <4 x i16>, <4 x i16>* %s
  %v2 = zext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_ext_v4i32_with_unfolded_offset:
; NO-SIMD128-NOT: load16x4
; SIMD128-NEXT: .functype load_ext_v4i32_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i16> @load_ext_v4i32_with_unfolded_offset(<4 x i16>* %p) {
  %q = ptrtoint <4 x i16>* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to <4 x i16>*
  %v = load <4 x i16>, <4 x i16>* %s
  ret <4 x i16> %v
}

; CHECK-LABEL: load_v4i32_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v4i32_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_v4i32_with_unfolded_gep_offset(<4 x i32>* %p) {
  %s = getelementptr <4 x i32>, <4 x i32>* %p, i32 1
  %v = load <4 x i32>, <4 x i32>* %s
  ret <4 x i32> %v
}

; CHECK-LABEL: load_splat_v4i32_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v4i32_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 4{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v32x4.load_splat $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_splat_v4i32_with_unfolded_gep_offset(i32* %p) {
  %s = getelementptr i32, i32* %p, i32 1
  %e = load i32, i32* %s
  %v1 = insertelement <4 x i32> undef, i32 %e, i32 0
  %v2 = shufflevector <4 x i32> %v1, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_sext_v4i32_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_sext_v4i32_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i32x4.load16x4_s $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_sext_v4i32_with_unfolded_gep_offset(<4 x i16>* %p) {
  %s = getelementptr <4 x i16>, <4 x i16>* %p, i32 1
  %v = load <4 x i16>, <4 x i16>* %s
  %v2 = sext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_zext_v4i32_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_zext_v4i32_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_zext_v4i32_with_unfolded_gep_offset(<4 x i16>* %p) {
  %s = getelementptr <4 x i16>, <4 x i16>* %p, i32 1
  %v = load <4 x i16>, <4 x i16>* %s
  %v2 = zext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_ext_v4i32_with_unfolded_gep_offset:
; NO-SIMD128-NOT: load16x4
; SIMD128-NEXT: .functype load_ext_v4i32_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i16> @load_ext_v4i32_with_unfolded_gep_offset(<4 x i16>* %p) {
  %s = getelementptr <4 x i16>, <4 x i16>* %p, i32 1
  %v = load <4 x i16>, <4 x i16>* %s
  ret <4 x i16> %v
}

; CHECK-LABEL: load_v4i32_from_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v4i32_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_v4i32_from_numeric_address() {
  %s = inttoptr i32 32 to <4 x i32>*
  %v = load <4 x i32>, <4 x i32>* %s
  ret <4 x i32> %v
}

; CHECK-LABEL: load_splat_v4i32_from_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v4i32_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v32x4.load_splat $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_splat_v4i32_from_numeric_address() {
  %s = inttoptr i32 32 to i32*
  %e = load i32, i32* %s
  %v1 = insertelement <4 x i32> undef, i32 %e, i32 0
  %v2 = shufflevector <4 x i32> %v1, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_sext_v4i32_from_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_sext_v4i32_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: i32x4.load16x4_s $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_sext_v4i32_from_numeric_address() {
  %s = inttoptr i32 32 to <4 x i16>*
  %v = load <4 x i16>, <4 x i16>* %s
  %v2 = sext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_zext_v4i32_from_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_zext_v4i32_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_zext_v4i32_from_numeric_address() {
  %s = inttoptr i32 32 to <4 x i16>*
  %v = load <4 x i16>, <4 x i16>* %s
  %v2 = zext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_ext_v4i32_from_numeric_address:
; NO-SIMD128-NOT: load16x4
; SIMD128-NEXT: .functype load_ext_v4i32_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i16> @load_ext_v4i32_from_numeric_address() {
  %s = inttoptr i32 32 to <4 x i16>*
  %v = load <4 x i16>, <4 x i16>* %s
  ret <4 x i16> %v
}

; CHECK-LABEL: load_v4i32_from_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v4i32_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, gv_v4i32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
@gv_v4i32 = global <4 x i32> <i32 42, i32 42, i32 42, i32 42>
define <4 x i32> @load_v4i32_from_global_address() {
  %v = load <4 x i32>, <4 x i32>* @gv_v4i32
  ret <4 x i32> %v
}

; CHECK-LABEL: load_splat_v4i32_from_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v4i32_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v32x4.load_splat $push[[R:[0-9]+]]=, gv_i32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
@gv_i32 = global i32 42
define <4 x i32> @load_splat_v4i32_from_global_address() {
  %e = load i32, i32* @gv_i32
  %v1 = insertelement <4 x i32> undef, i32 %e, i32 0
  %v2 = shufflevector <4 x i32> %v1, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_sext_v4i32_from_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_sext_v4i32_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: i32x4.load16x4_s $push[[R:[0-9]+]]=, gv_v4i16($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
@gv_v4i16 = global <4 x i16> <i16 42, i16 42, i16 42, i16 42>
define <4 x i32> @load_sext_v4i32_from_global_address() {
  %v = load <4 x i16>, <4 x i16>* @gv_v4i16
  %v2 = sext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_zext_v4i32_from_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_zext_v4i32_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, gv_v4i16($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @load_zext_v4i32_from_global_address() {
  %v = load <4 x i16>, <4 x i16>* @gv_v4i16
  %v2 = zext <4 x i16> %v to <4 x i32>
  ret <4 x i32> %v2
}

; CHECK-LABEL: load_ext_v4i32_from_global_address:
; NO-SIMD128-NOT: load16x4
; SIMD128-NEXT: .functype load_ext_v4i32_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: i32x4.load16x4_u $push[[R:[0-9]+]]=, gv_v4i16($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i16> @load_ext_v4i32_from_global_address() {
  %v = load <4 x i16>, <4 x i16>* @gv_v4i16
  ret <4 x i16> %v
}

; CHECK-LABEL: store_v4i32:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v4i32 (v128, i32) -> (){{$}}
; SIMD128-NEXT: v128.store 0($1), $0{{$}}
define void @store_v4i32(<4 x i32> %v, <4 x i32>* %p) {
  store <4 x i32> %v , <4 x i32>* %p
  ret void
}

; CHECK-LABEL: store_v4i32_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v4i32_with_folded_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: v128.store 16($1), $0{{$}}
define void @store_v4i32_with_folded_offset(<4 x i32> %v, <4 x i32>* %p) {
  %q = ptrtoint <4 x i32>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <4 x i32>*
  store <4 x i32> %v , <4 x i32>* %s
  ret void
}

; CHECK-LABEL: store_v4i32_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v4i32_with_folded_gep_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: v128.store 16($1), $0{{$}}
define void @store_v4i32_with_folded_gep_offset(<4 x i32> %v, <4 x i32>* %p) {
  %s = getelementptr inbounds <4 x i32>, <4 x i32>* %p, i32 1
  store <4 x i32> %v , <4 x i32>* %s
  ret void
}

; CHECK-LABEL: store_v4i32_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v4i32_with_unfolded_gep_negative_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -16{{$}}
; SIMD128-NEXT: i32.add $push[[R:[0-9]+]]=, $1, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[R]]), $0{{$}}
define void @store_v4i32_with_unfolded_gep_negative_offset(<4 x i32> %v, <4 x i32>* %p) {
  %s = getelementptr inbounds <4 x i32>, <4 x i32>* %p, i32 -1
  store <4 x i32> %v , <4 x i32>* %s
  ret void
}

; CHECK-LABEL: store_v4i32_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v4i32_with_unfolded_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -16{{$}}
; SIMD128-NEXT: i32.add $push[[R:[0-9]+]]=, $1, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[R]]), $0{{$}}
define void @store_v4i32_with_unfolded_offset(<4 x i32> %v, <4 x i32>* %p) {
  %s = getelementptr inbounds <4 x i32>, <4 x i32>* %p, i32 -1
  store <4 x i32> %v , <4 x i32>* %s
  ret void
}

; CHECK-LABEL: store_v4i32_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v4i32_with_unfolded_gep_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[R:[0-9]+]]=, $1, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[R]]), $0{{$}}
define void @store_v4i32_with_unfolded_gep_offset(<4 x i32> %v, <4 x i32>* %p) {
  %s = getelementptr <4 x i32>, <4 x i32>* %p, i32 1
  store <4 x i32> %v , <4 x i32>* %s
  ret void
}

; CHECK-LABEL: store_v4i32_to_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v4i32_to_numeric_address (v128) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.store 32($pop[[L0]]), $0{{$}}
define void @store_v4i32_to_numeric_address(<4 x i32> %v) {
  %s = inttoptr i32 32 to <4 x i32>*
  store <4 x i32> %v , <4 x i32>* %s
  ret void
}

; CHECK-LABEL: store_v4i32_to_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v4i32_to_global_address (v128) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[R:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.store gv_v4i32($pop[[R]]), $0{{$}}
define void @store_v4i32_to_global_address(<4 x i32> %v) {
  store <4 x i32> %v , <4 x i32>* @gv_v4i32
  ret void
}

; ==============================================================================
; 2 x i64
; ==============================================================================
; CHECK-LABEL: load_v2i64:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_v2i64 (i32) -> (v128){{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_v2i64(<2 x i64>* %p) {
  %v = load <2 x i64>, <2 x i64>* %p
  ret <2 x i64> %v
}

; CHECK-LABEL: load_splat_v2i64:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_splat_v2i64 (i32) -> (v128){{$}}
; SIMD128-NEXT: v64x2.load_splat $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_splat_v2i64(i64* %p) {
  %e = load i64, i64* %p
  %v1 = insertelement <2 x i64> undef, i64 %e, i32 0
  %v2 = shufflevector <2 x i64> %v1, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_sext_v2i64:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_sext_v2i64 (i32) -> (v128){{$}}
; SIMD128-NEXT: i64x2.load32x2_s $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_sext_v2i64(<2 x i32>* %p) {
  %v = load <2 x i32>, <2 x i32>* %p
  %v2 = sext <2 x i32> %v to <2 x i64>
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_zext_v2i64:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_zext_v2i64 (i32) -> (v128){{$}}
; SIMD128-NEXT: i64x2.load32x2_u $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_zext_v2i64(<2 x i32>* %p) {
  %v = load <2 x i32>, <2 x i32>* %p
  %v2 = zext <2 x i32> %v to <2 x i64>
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_ext_v2i64:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: load32x2
; SIMD128-NEXT: .functype load_ext_v2i64 (i32) -> (v128){{$}}
; SIMD128-NEXT: i64x2.load32x2_u $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i32> @load_ext_v2i64(<2 x i32>* %p) {
  %v = load <2 x i32>, <2 x i32>* %p
  ret <2 x i32> %v
}

; CHECK-LABEL: load_v2i64_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_v2i64_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_v2i64_with_folded_offset(<2 x i64>* %p) {
  %q = ptrtoint <2 x i64>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <2 x i64>*
  %v = load <2 x i64>, <2 x i64>* %s
  ret <2 x i64> %v
}

; CHECK-LABEL: load_splat_v2i64_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_splat_v2i64_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v64x2.load_splat $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_splat_v2i64_with_folded_offset(i64* %p) {
  %q = ptrtoint i64* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to i64*
  %e = load i64, i64* %s
  %v1 = insertelement <2 x i64> undef, i64 %e, i32 0
  %v2 = shufflevector <2 x i64> %v1, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_sext_v2i64_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_sext_v2i64_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i64x2.load32x2_s $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_sext_v2i64_with_folded_offset(<2 x i32>* %p) {
  %q = ptrtoint <2 x i32>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <2 x i32>*
  %v = load <2 x i32>, <2 x i32>* %s
  %v2 = sext <2 x i32> %v to <2 x i64>
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_zext_v2i64_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_zext_v2i64_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i64x2.load32x2_u $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_zext_v2i64_with_folded_offset(<2 x i32>* %p) {
  %q = ptrtoint <2 x i32>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <2 x i32>*
  %v = load <2 x i32>, <2 x i32>* %s
  %v2 = zext <2 x i32> %v to <2 x i64>
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_ext_v2i64_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: load32x2
; SIMD128-NEXT: .functype load_ext_v2i64_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i64x2.load32x2_u $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i32> @load_ext_v2i64_with_folded_offset(<2 x i32>* %p) {
  %q = ptrtoint <2 x i32>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <2 x i32>*
  %v = load <2 x i32>, <2 x i32>* %s
  ret <2 x i32> %v
}

; CHECK-LABEL: load_v2i64_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_v2i64_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_v2i64_with_folded_gep_offset(<2 x i64>* %p) {
  %s = getelementptr inbounds <2 x i64>, <2 x i64>* %p, i32 1
  %v = load <2 x i64>, <2 x i64>* %s
  ret <2 x i64> %v
}

; CHECK-LABEL: load_splat_v2i64_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_splat_v2i64_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v64x2.load_splat $push[[R:[0-9]+]]=, 8($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_splat_v2i64_with_folded_gep_offset(i64* %p) {
  %s = getelementptr inbounds i64, i64* %p, i32 1
  %e = load i64, i64* %s
  %v1 = insertelement <2 x i64> undef, i64 %e, i32 0
  %v2 = shufflevector <2 x i64> %v1, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_sext_v2i64_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_sext_v2i64_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i64x2.load32x2_s $push[[R:[0-9]+]]=, 8($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_sext_v2i64_with_folded_gep_offset(<2 x i32>* %p) {
  %s = getelementptr inbounds <2 x i32>, <2 x i32>* %p, i32 1
  %v = load <2 x i32>, <2 x i32>* %s
  %v2 = sext <2 x i32> %v to <2 x i64>
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_zext_v2i64_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_zext_v2i64_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i64x2.load32x2_u $push[[R:[0-9]+]]=, 8($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_zext_v2i64_with_folded_gep_offset(<2 x i32>* %p) {
  %s = getelementptr inbounds <2 x i32>, <2 x i32>* %p, i32 1
  %v = load <2 x i32>, <2 x i32>* %s
  %v2 = zext <2 x i32> %v to <2 x i64>
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_ext_v2i64_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: load32x2
; SIMD128-NEXT: .functype load_ext_v2i64_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i64x2.load32x2_u $push[[R:[0-9]+]]=, 8($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i32> @load_ext_v2i64_with_folded_gep_offset(<2 x i32>* %p) {
  %s = getelementptr inbounds <2 x i32>, <2 x i32>* %p, i32 1
  %v = load <2 x i32>, <2 x i32>* %s
  ret <2 x i32> %v
}

; CHECK-LABEL: load_v2i64_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_v2i64_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_v2i64_with_unfolded_gep_negative_offset(<2 x i64>* %p) {
  %s = getelementptr inbounds <2 x i64>, <2 x i64>* %p, i32 -1
  %v = load <2 x i64>, <2 x i64>* %s
  ret <2 x i64> %v
}

; CHECK-LABEL: load_splat_v2i64_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_splat_v2i64_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v64x2.load_splat $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_splat_v2i64_with_unfolded_gep_negative_offset(i64* %p) {
  %s = getelementptr inbounds i64, i64* %p, i32 -1
  %e = load i64, i64* %s
  %v1 = insertelement <2 x i64> undef, i64 %e, i32 0
  %v2 = shufflevector <2 x i64> %v1, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_sext_v2i64_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_sext_v2i64_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i64x2.load32x2_s $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_sext_v2i64_with_unfolded_gep_negative_offset(<2 x i32>* %p) {
  %s = getelementptr inbounds <2 x i32>, <2 x i32>* %p, i32 -1
  %v = load <2 x i32>, <2 x i32>* %s
  %v2 = sext <2 x i32> %v to <2 x i64>
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_zext_v2i64_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_zext_v2i64_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i64x2.load32x2_u $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_zext_v2i64_with_unfolded_gep_negative_offset(<2 x i32>* %p) {
  %s = getelementptr inbounds <2 x i32>, <2 x i32>* %p, i32 -1
  %v = load <2 x i32>, <2 x i32>* %s
  %v2 = zext <2 x i32> %v to <2 x i64>
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_ext_v2i64_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: load32x2
; SIMD128-NEXT: .functype load_ext_v2i64_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i64x2.load32x2_u $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i32> @load_ext_v2i64_with_unfolded_gep_negative_offset(<2 x i32>* %p) {
  %s = getelementptr inbounds <2 x i32>, <2 x i32>* %p, i32 -1
  %v = load <2 x i32>, <2 x i32>* %s
  ret <2 x i32> %v
}

; CHECK-LABEL: load_v2i64_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_v2i64_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_v2i64_with_unfolded_offset(<2 x i64>* %p) {
  %q = ptrtoint <2 x i64>* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to <2 x i64>*
  %v = load <2 x i64>, <2 x i64>* %s
  ret <2 x i64> %v
}

; CHECK-LABEL: load_splat_v2i64_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_splat_v2i64_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v64x2.load_splat $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_splat_v2i64_with_unfolded_offset(i64* %p) {
  %q = ptrtoint i64* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to i64*
  %e = load i64, i64* %s
  %v1 = insertelement <2 x i64> undef, i64 %e, i32 0
  %v2 = shufflevector <2 x i64> %v1, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_sext_v2i64_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_sext_v2i64_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i64x2.load32x2_s $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_sext_v2i64_with_unfolded_offset(<2 x i32>* %p) {
  %q = ptrtoint <2 x i32>* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to <2 x i32>*
  %v = load <2 x i32>, <2 x i32>* %s
  %v2 = sext <2 x i32> %v to <2 x i64>
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_zext_v2i64_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_zext_v2i64_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i64x2.load32x2_u $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_zext_v2i64_with_unfolded_offset(<2 x i32>* %p) {
  %q = ptrtoint <2 x i32>* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to <2 x i32>*
  %v = load <2 x i32>, <2 x i32>* %s
  %v2 = zext <2 x i32> %v to <2 x i64>
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_ext_v2i64_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: load32x2
; SIMD128-NEXT: .functype load_ext_v2i64_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i64x2.load32x2_u $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i32> @load_ext_v2i64_with_unfolded_offset(<2 x i32>* %p) {
  %q = ptrtoint <2 x i32>* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to <2 x i32>*
  %v = load <2 x i32>, <2 x i32>* %s
  ret <2 x i32> %v
}

; CHECK-LABEL: load_v2i64_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_v2i64_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_v2i64_with_unfolded_gep_offset(<2 x i64>* %p) {
  %s = getelementptr <2 x i64>, <2 x i64>* %p, i32 1
  %v = load <2 x i64>, <2 x i64>* %s
  ret <2 x i64> %v
}

; CHECK-LABEL: load_splat_v2i64_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_splat_v2i64_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v64x2.load_splat $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_splat_v2i64_with_unfolded_gep_offset(i64* %p) {
  %s = getelementptr i64, i64* %p, i32 1
  %e = load i64, i64* %s
  %v1 = insertelement <2 x i64> undef, i64 %e, i32 0
  %v2 = shufflevector <2 x i64> %v1, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_sext_v2i64_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_sext_v2i64_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i64x2.load32x2_s $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_sext_v2i64_with_unfolded_gep_offset(<2 x i32>* %p) {
  %s = getelementptr <2 x i32>, <2 x i32>* %p, i32 1
  %v = load <2 x i32>, <2 x i32>* %s
  %v2 = sext <2 x i32> %v to <2 x i64>
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_zext_v2i64_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_zext_v2i64_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i64x2.load32x2_u $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_zext_v2i64_with_unfolded_gep_offset(<2 x i32>* %p) {
  %s = getelementptr <2 x i32>, <2 x i32>* %p, i32 1
  %v = load <2 x i32>, <2 x i32>* %s
  %v2 = zext <2 x i32> %v to <2 x i64>
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_ext_v2i64_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: load32x2
; SIMD128-NEXT: .functype load_ext_v2i64_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: i64x2.load32x2_u $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i32> @load_ext_v2i64_with_unfolded_gep_offset(<2 x i32>* %p) {
  %s = getelementptr <2 x i32>, <2 x i32>* %p, i32 1
  %v = load <2 x i32>, <2 x i32>* %s
  ret <2 x i32> %v
}

; CHECK-LABEL: load_v2i64_from_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_v2i64_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_v2i64_from_numeric_address() {
  %s = inttoptr i32 32 to <2 x i64>*
  %v = load <2 x i64>, <2 x i64>* %s
  ret <2 x i64> %v
}

; CHECK-LABEL: load_splat_v2i64_from_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_splat_v2i64_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v64x2.load_splat $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_splat_v2i64_from_numeric_address() {
  %s = inttoptr i32 32 to i64*
  %e = load i64, i64* %s
  %v1 = insertelement <2 x i64> undef, i64 %e, i32 0
  %v2 = shufflevector <2 x i64> %v1, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_sext_v2i64_from_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_sext_v2i64_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: i64x2.load32x2_s $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_sext_v2i64_from_numeric_address() {
  %s = inttoptr i32 32 to <2 x i32>*
  %v = load <2 x i32>, <2 x i32>* %s
  %v2 = sext <2 x i32> %v to <2 x i64>
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_zext_v2i64_from_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_zext_v2i64_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: i64x2.load32x2_u $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_zext_v2i64_from_numeric_address() {
  %s = inttoptr i32 32 to <2 x i32>*
  %v = load <2 x i32>, <2 x i32>* %s
  %v2 = zext <2 x i32> %v to <2 x i64>
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_ext_v2i64_from_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: load32x2
; SIMD128-NEXT: .functype load_ext_v2i64_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: i64x2.load32x2_u $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i32> @load_ext_v2i64_from_numeric_address() {
  %s = inttoptr i32 32 to <2 x i32>*
  %v = load <2 x i32>, <2 x i32>* %s
  ret <2 x i32> %v
}

; CHECK-LABEL: load_v2i64_from_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_v2i64_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, gv_v2i64($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
@gv_v2i64 = global <2 x i64> <i64 42, i64 42>
define <2 x i64> @load_v2i64_from_global_address() {
  %v = load <2 x i64>, <2 x i64>* @gv_v2i64
  ret <2 x i64> %v
}

; CHECK-LABEL: load_splat_v2i64_from_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_splat_v2i64_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v64x2.load_splat $push[[R:[0-9]+]]=, gv_i64($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
@gv_i64 = global i64 42
define <2 x i64> @load_splat_v2i64_from_global_address() {
  %e = load i64, i64* @gv_i64
  %v1 = insertelement <2 x i64> undef, i64 %e, i32 0
  %v2 = shufflevector <2 x i64> %v1, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_sext_v2i64_from_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_sext_v2i64_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: i64x2.load32x2_s $push[[R:[0-9]+]]=, gv_v2i32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
@gv_v2i32 = global <2 x i32> <i32 42, i32 42>
define <2 x i64> @load_sext_v2i64_from_global_address() {
  %v = load <2 x i32>, <2 x i32>* @gv_v2i32
  %v2 = sext <2 x i32> %v to <2 x i64>
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_zext_v2i64_from_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_zext_v2i64_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: i64x2.load32x2_u $push[[R:[0-9]+]]=, gv_v2i32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_zext_v2i64_from_global_address() {
  %v = load <2 x i32>, <2 x i32>* @gv_v2i32
  %v2 = zext <2 x i32> %v to <2 x i64>
  ret <2 x i64> %v2
}

; CHECK-LABEL: load_ext_v2i64_from_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: load32x2
; SIMD128-NEXT: .functype load_ext_v2i64_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: i64x2.load32x2_u $push[[R:[0-9]+]]=, gv_v2i32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i32> @load_ext_v2i64_from_global_address() {
  %v = load <2 x i32>, <2 x i32>* @gv_v2i32
  ret <2 x i32> %v
}

; CHECK-LABEL: store_v2i64:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype store_v2i64 (v128, i32) -> (){{$}}
; SIMD128-NEXT: v128.store 0($1), $0{{$}}
define void @store_v2i64(<2 x i64> %v, <2 x i64>* %p) {
  store <2 x i64> %v , <2 x i64>* %p
  ret void
}

; CHECK-LABEL: store_v2i64_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype store_v2i64_with_folded_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: v128.store 16($1), $0{{$}}
define void @store_v2i64_with_folded_offset(<2 x i64> %v, <2 x i64>* %p) {
  %q = ptrtoint <2 x i64>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <2 x i64>*
  store <2 x i64> %v , <2 x i64>* %s
  ret void
}

; CHECK-LABEL: store_v2i64_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype store_v2i64_with_folded_gep_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: v128.store 16($1), $0{{$}}
define void @store_v2i64_with_folded_gep_offset(<2 x i64> %v, <2 x i64>* %p) {
  %s = getelementptr inbounds <2 x i64>, <2 x i64>* %p, i32 1
  store <2 x i64> %v , <2 x i64>* %s
  ret void
}

; CHECK-LABEL: store_v2i64_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype store_v2i64_with_unfolded_gep_negative_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -16{{$}}
; SIMD128-NEXT: i32.add $push[[R:[0-9]+]]=, $1, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[R]]), $0{{$}}
define void @store_v2i64_with_unfolded_gep_negative_offset(<2 x i64> %v, <2 x i64>* %p) {
  %s = getelementptr inbounds <2 x i64>, <2 x i64>* %p, i32 -1
  store <2 x i64> %v , <2 x i64>* %s
  ret void
}

; CHECK-LABEL: store_v2i64_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype store_v2i64_with_unfolded_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -16{{$}}
; SIMD128-NEXT: i32.add $push[[R:[0-9]+]]=, $1, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[R]]), $0{{$}}
define void @store_v2i64_with_unfolded_offset(<2 x i64> %v, <2 x i64>* %p) {
  %s = getelementptr inbounds <2 x i64>, <2 x i64>* %p, i32 -1
  store <2 x i64> %v , <2 x i64>* %s
  ret void
}

; CHECK-LABEL: store_v2i64_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype store_v2i64_with_unfolded_gep_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[R:[0-9]+]]=, $1, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[R]]), $0{{$}}
define void @store_v2i64_with_unfolded_gep_offset(<2 x i64> %v, <2 x i64>* %p) {
  %s = getelementptr <2 x i64>, <2 x i64>* %p, i32 1
  store <2 x i64> %v , <2 x i64>* %s
  ret void
}

; CHECK-LABEL: store_v2i64_to_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype store_v2i64_to_numeric_address (v128) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.store 32($pop[[L0]]), $0{{$}}
define void @store_v2i64_to_numeric_address(<2 x i64> %v) {
  %s = inttoptr i32 32 to <2 x i64>*
  store <2 x i64> %v , <2 x i64>* %s
  ret void
}

; CHECK-LABEL: store_v2i64_to_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype store_v2i64_to_global_address (v128) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[R:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.store gv_v2i64($pop[[R]]), $0{{$}}
define void @store_v2i64_to_global_address(<2 x i64> %v) {
  store <2 x i64> %v , <2 x i64>* @gv_v2i64
  ret void
}

; ==============================================================================
; 4 x float
; ==============================================================================
; CHECK-LABEL: load_v4f32:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v4f32 (i32) -> (v128){{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @load_v4f32(<4 x float>* %p) {
  %v = load <4 x float>, <4 x float>* %p
  ret <4 x float> %v
}

; CHECK-LABEL: load_splat_v4f32:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v4f32 (i32) -> (v128){{$}}
; SIMD128-NEXT: v32x4.load_splat $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @load_splat_v4f32(float* %p) {
  %e = load float, float* %p
  %v1 = insertelement <4 x float> undef, float %e, i32 0
  %v2 = shufflevector <4 x float> %v1, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %v2
}

; CHECK-LABEL: load_v4f32_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v4f32_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @load_v4f32_with_folded_offset(<4 x float>* %p) {
  %q = ptrtoint <4 x float>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <4 x float>*
  %v = load <4 x float>, <4 x float>* %s
  ret <4 x float> %v
}

; CHECK-LABEL: load_splat_v4f32_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v4f32_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v32x4.load_splat $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @load_splat_v4f32_with_folded_offset(float* %p) {
  %q = ptrtoint float* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to float*
  %e = load float, float* %s
  %v1 = insertelement <4 x float> undef, float %e, i32 0
  %v2 = shufflevector <4 x float> %v1, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %v2
}

; CHECK-LABEL: load_v4f32_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v4f32_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @load_v4f32_with_folded_gep_offset(<4 x float>* %p) {
  %s = getelementptr inbounds <4 x float>, <4 x float>* %p, i32 1
  %v = load <4 x float>, <4 x float>* %s
  ret <4 x float> %v
}

; CHECK-LABEL: load_splat_v4f32_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v4f32_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v32x4.load_splat $push[[R:[0-9]+]]=, 4($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @load_splat_v4f32_with_folded_gep_offset(float* %p) {
  %s = getelementptr inbounds float, float* %p, i32 1
  %e = load float, float* %s
  %v1 = insertelement <4 x float> undef, float %e, i32 0
  %v2 = shufflevector <4 x float> %v1, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %v2
}

; CHECK-LABEL: load_v4f32_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v4f32_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @load_v4f32_with_unfolded_gep_negative_offset(<4 x float>* %p) {
  %s = getelementptr inbounds <4 x float>, <4 x float>* %p, i32 -1
  %v = load <4 x float>, <4 x float>* %s
  ret <4 x float> %v
}

; CHECK-LABEL: load_splat_v4f32_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v4f32_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -4{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v32x4.load_splat $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @load_splat_v4f32_with_unfolded_gep_negative_offset(float* %p) {
  %s = getelementptr inbounds float, float* %p, i32 -1
  %e = load float, float* %s
  %v1 = insertelement <4 x float> undef, float %e, i32 0
  %v2 = shufflevector <4 x float> %v1, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %v2
}

; CHECK-LABEL: load_v4f32_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v4f32_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @load_v4f32_with_unfolded_offset(<4 x float>* %p) {
  %q = ptrtoint <4 x float>* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to <4 x float>*
  %v = load <4 x float>, <4 x float>* %s
  ret <4 x float> %v
}

; CHECK-LABEL: load_splat_v4f32_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v4f32_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v32x4.load_splat $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @load_splat_v4f32_with_unfolded_offset(float* %p) {
  %q = ptrtoint float* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to float*
  %e = load float, float* %s
  %v1 = insertelement <4 x float> undef, float %e, i32 0
  %v2 = shufflevector <4 x float> %v1, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %v2
}

; CHECK-LABEL: load_v4f32_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v4f32_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @load_v4f32_with_unfolded_gep_offset(<4 x float>* %p) {
  %s = getelementptr <4 x float>, <4 x float>* %p, i32 1
  %v = load <4 x float>, <4 x float>* %s
  ret <4 x float> %v
}

; CHECK-LABEL: load_splat_v4f32_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v4f32_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 4{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v32x4.load_splat $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @load_splat_v4f32_with_unfolded_gep_offset(float* %p) {
  %s = getelementptr float, float* %p, i32 1
  %e = load float, float* %s
  %v1 = insertelement <4 x float> undef, float %e, i32 0
  %v2 = shufflevector <4 x float> %v1, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %v2
}

; CHECK-LABEL: load_v4f32_from_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v4f32_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @load_v4f32_from_numeric_address() {
  %s = inttoptr i32 32 to <4 x float>*
  %v = load <4 x float>, <4 x float>* %s
  ret <4 x float> %v
}

; CHECK-LABEL: load_splat_v4f32_from_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v4f32_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v32x4.load_splat $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @load_splat_v4f32_from_numeric_address() {
  %s = inttoptr i32 32 to float*
  %e = load float, float* %s
  %v1 = insertelement <4 x float> undef, float %e, i32 0
  %v2 = shufflevector <4 x float> %v1, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %v2
}

; CHECK-LABEL: load_v4f32_from_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_v4f32_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, gv_v4f32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
@gv_v4f32 = global <4 x float> <float 42., float 42., float 42., float 42.>
define <4 x float> @load_v4f32_from_global_address() {
  %v = load <4 x float>, <4 x float>* @gv_v4f32
  ret <4 x float> %v
}

; CHECK-LABEL: load_splat_v4f32_from_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype load_splat_v4f32_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v32x4.load_splat $push[[R:[0-9]+]]=, gv_f32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
@gv_f32 = global float 42.
define <4 x float> @load_splat_v4f32_from_global_address() {
  %e = load float, float* @gv_f32
  %v1 = insertelement <4 x float> undef, float %e, i32 0
  %v2 = shufflevector <4 x float> %v1, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %v2
}

; CHECK-LABEL: store_v4f32:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v4f32 (v128, i32) -> (){{$}}
; SIMD128-NEXT: v128.store 0($1), $0{{$}}
define void @store_v4f32(<4 x float> %v, <4 x float>* %p) {
  store <4 x float> %v , <4 x float>* %p
  ret void
}

; CHECK-LABEL: store_v4f32_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v4f32_with_folded_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: v128.store 16($1), $0{{$}}
define void @store_v4f32_with_folded_offset(<4 x float> %v, <4 x float>* %p) {
  %q = ptrtoint <4 x float>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <4 x float>*
  store <4 x float> %v , <4 x float>* %s
  ret void
}

; CHECK-LABEL: store_v4f32_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v4f32_with_folded_gep_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: v128.store 16($1), $0{{$}}
define void @store_v4f32_with_folded_gep_offset(<4 x float> %v, <4 x float>* %p) {
  %s = getelementptr inbounds <4 x float>, <4 x float>* %p, i32 1
  store <4 x float> %v , <4 x float>* %s
  ret void
}

; CHECK-LABEL: store_v4f32_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v4f32_with_unfolded_gep_negative_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -16{{$}}
; SIMD128-NEXT: i32.add $push[[R:[0-9]+]]=, $1, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[R]]), $0{{$}}
define void @store_v4f32_with_unfolded_gep_negative_offset(<4 x float> %v, <4 x float>* %p) {
  %s = getelementptr inbounds <4 x float>, <4 x float>* %p, i32 -1
  store <4 x float> %v , <4 x float>* %s
  ret void
}

; CHECK-LABEL: store_v4f32_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v4f32_with_unfolded_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -16{{$}}
; SIMD128-NEXT: i32.add $push[[R:[0-9]+]]=, $1, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[R]]), $0{{$}}
define void @store_v4f32_with_unfolded_offset(<4 x float> %v, <4 x float>* %p) {
  %s = getelementptr inbounds <4 x float>, <4 x float>* %p, i32 -1
  store <4 x float> %v , <4 x float>* %s
  ret void
}

; CHECK-LABEL: store_v4f32_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v4f32_with_unfolded_gep_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[R:[0-9]+]]=, $1, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[R]]), $0{{$}}
define void @store_v4f32_with_unfolded_gep_offset(<4 x float> %v, <4 x float>* %p) {
  %s = getelementptr <4 x float>, <4 x float>* %p, i32 1
  store <4 x float> %v , <4 x float>* %s
  ret void
}

; CHECK-LABEL: store_v4f32_to_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v4f32_to_numeric_address (v128) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.store 32($pop[[L0]]), $0{{$}}
define void @store_v4f32_to_numeric_address(<4 x float> %v) {
  %s = inttoptr i32 32 to <4 x float>*
  store <4 x float> %v , <4 x float>* %s
  ret void
}

; CHECK-LABEL: store_v4f32_to_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype store_v4f32_to_global_address (v128) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[R:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.store gv_v4f32($pop[[R]]), $0{{$}}
define void @store_v4f32_to_global_address(<4 x float> %v) {
  store <4 x float> %v , <4 x float>* @gv_v4f32
  ret void
}

; ==============================================================================
; 2 x double
; ==============================================================================
; CHECK-LABEL: load_v2f64:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_v2f64 (i32) -> (v128){{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @load_v2f64(<2 x double>* %p) {
  %v = load <2 x double>, <2 x double>* %p
  ret <2 x double> %v
}

; CHECK-LABEL: load_splat_v2f64:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_splat_v2f64 (i32) -> (v128){{$}}
; SIMD128-NEXT: v64x2.load_splat $push[[R:[0-9]+]]=, 0($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @load_splat_v2f64(double* %p) {
  %e = load double, double* %p
  %v1 = insertelement <2 x double> undef, double %e, i32 0
  %v2 = shufflevector <2 x double> %v1, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %v2
}

; CHECK-LABEL: load_v2f64_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_v2f64_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @load_v2f64_with_folded_offset(<2 x double>* %p) {
  %q = ptrtoint <2 x double>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <2 x double>*
  %v = load <2 x double>, <2 x double>* %s
  ret <2 x double> %v
}

; CHECK-LABEL: load_splat_v2f64_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_splat_v2f64_with_folded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v64x2.load_splat $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @load_splat_v2f64_with_folded_offset(double* %p) {
  %q = ptrtoint double* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to double*
  %e = load double, double* %s
  %v1 = insertelement <2 x double> undef, double %e, i32 0
  %v2 = shufflevector <2 x double> %v1, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %v2
}

; CHECK-LABEL: load_v2f64_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_v2f64_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 16($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @load_v2f64_with_folded_gep_offset(<2 x double>* %p) {
  %s = getelementptr inbounds <2 x double>, <2 x double>* %p, i32 1
  %v = load <2 x double>, <2 x double>* %s
  ret <2 x double> %v
}

; CHECK-LABEL: load_splat_v2f64_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_splat_v2f64_with_folded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: v64x2.load_splat $push[[R:[0-9]+]]=, 8($0){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @load_splat_v2f64_with_folded_gep_offset(double* %p) {
  %s = getelementptr inbounds double, double* %p, i32 1
  %e = load double, double* %s
  %v1 = insertelement <2 x double> undef, double %e, i32 0
  %v2 = shufflevector <2 x double> %v1, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %v2
}

; CHECK-LABEL: load_v2f64_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_v2f64_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @load_v2f64_with_unfolded_gep_negative_offset(<2 x double>* %p) {
  %s = getelementptr inbounds <2 x double>, <2 x double>* %p, i32 -1
  %v = load <2 x double>, <2 x double>* %s
  ret <2 x double> %v
}

; CHECK-LABEL: load_splat_v2f64_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_splat_v2f64_with_unfolded_gep_negative_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v64x2.load_splat $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @load_splat_v2f64_with_unfolded_gep_negative_offset(double* %p) {
  %s = getelementptr inbounds double, double* %p, i32 -1
  %e = load double, double* %s
  %v1 = insertelement <2 x double> undef, double %e, i32 0
  %v2 = shufflevector <2 x double> %v1, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %v2
}

; CHECK-LABEL: load_v2f64_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_v2f64_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @load_v2f64_with_unfolded_offset(<2 x double>* %p) {
  %q = ptrtoint <2 x double>* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to <2 x double>*
  %v = load <2 x double>, <2 x double>* %s
  ret <2 x double> %v
}

; CHECK-LABEL: load_splat_v2f64_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_splat_v2f64_with_unfolded_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v64x2.load_splat $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @load_splat_v2f64_with_unfolded_offset(double* %p) {
  %q = ptrtoint double* %p to i32
  %r = add nsw i32 %q, 16
  %s = inttoptr i32 %r to double*
  %e = load double, double* %s
  %v1 = insertelement <2 x double> undef, double %e, i32 0
  %v2 = shufflevector <2 x double> %v1, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %v2
}

; CHECK-LABEL: load_v2f64_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_v2f64_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @load_v2f64_with_unfolded_gep_offset(<2 x double>* %p) {
  %s = getelementptr <2 x double>, <2 x double>* %p, i32 1
  %v = load <2 x double>, <2 x double>* %s
  ret <2 x double> %v
}

; CHECK-LABEL: load_splat_v2f64_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_splat_v2f64_with_unfolded_gep_offset (i32) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 8{{$}}
; SIMD128-NEXT: i32.add $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: v64x2.load_splat $push[[R:[0-9]+]]=, 0($pop[[L1]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @load_splat_v2f64_with_unfolded_gep_offset(double* %p) {
  %s = getelementptr double, double* %p, i32 1
  %e = load double, double* %s
  %v1 = insertelement <2 x double> undef, double %e, i32 0
  %v2 = shufflevector <2 x double> %v1, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %v2
}

; CHECK-LABEL: load_v2f64_from_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_v2f64_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @load_v2f64_from_numeric_address() {
  %s = inttoptr i32 32 to <2 x double>*
  %v = load <2 x double>, <2 x double>* %s
  ret <2 x double> %v
}

; CHECK-LABEL: load_splat_v2f64_from_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_splat_v2f64_from_numeric_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v64x2.load_splat $push[[R:[0-9]+]]=, 32($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @load_splat_v2f64_from_numeric_address() {
  %s = inttoptr i32 32 to double*
  %e = load double, double* %s
  %v1 = insertelement <2 x double> undef, double %e, i32 0
  %v2 = shufflevector <2 x double> %v1, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %v2
}

; CHECK-LABEL: load_v2f64_from_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_v2f64_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.load $push[[R:[0-9]+]]=, gv_v2f64($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
@gv_v2f64 = global <2 x double> <double 42., double 42.>
define <2 x double> @load_v2f64_from_global_address() {
  %v = load <2 x double>, <2 x double>* @gv_v2f64
  ret <2 x double> %v
}

; CHECK-LABEL: load_splat_v2f64_from_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype load_splat_v2f64_from_global_address () -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v64x2.load_splat $push[[R:[0-9]+]]=, gv_f64($pop[[L0]]){{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
@gv_f64 = global double 42.
define <2 x double> @load_splat_v2f64_from_global_address() {
  %e = load double, double* @gv_f64
  %v1 = insertelement <2 x double> undef, double %e, i32 0
  %v2 = shufflevector <2 x double> %v1, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %v2
}

; CHECK-LABEL: store_v2f64:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype store_v2f64 (v128, i32) -> (){{$}}
; SIMD128-NEXT: v128.store 0($1), $0{{$}}
define void @store_v2f64(<2 x double> %v, <2 x double>* %p) {
  store <2 x double> %v , <2 x double>* %p
  ret void
}

; CHECK-LABEL: store_v2f64_with_folded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype store_v2f64_with_folded_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: v128.store 16($1), $0{{$}}
define void @store_v2f64_with_folded_offset(<2 x double> %v, <2 x double>* %p) {
  %q = ptrtoint <2 x double>* %p to i32
  %r = add nuw i32 %q, 16
  %s = inttoptr i32 %r to <2 x double>*
  store <2 x double> %v , <2 x double>* %s
  ret void
}

; CHECK-LABEL: store_v2f64_with_folded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype store_v2f64_with_folded_gep_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: v128.store 16($1), $0{{$}}
define void @store_v2f64_with_folded_gep_offset(<2 x double> %v, <2 x double>* %p) {
  %s = getelementptr inbounds <2 x double>, <2 x double>* %p, i32 1
  store <2 x double> %v , <2 x double>* %s
  ret void
}

; CHECK-LABEL: store_v2f64_with_unfolded_gep_negative_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype store_v2f64_with_unfolded_gep_negative_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -16{{$}}
; SIMD128-NEXT: i32.add $push[[R:[0-9]+]]=, $1, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[R]]), $0{{$}}
define void @store_v2f64_with_unfolded_gep_negative_offset(<2 x double> %v, <2 x double>* %p) {
  %s = getelementptr inbounds <2 x double>, <2 x double>* %p, i32 -1
  store <2 x double> %v , <2 x double>* %s
  ret void
}

; CHECK-LABEL: store_v2f64_with_unfolded_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype store_v2f64_with_unfolded_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, -16{{$}}
; SIMD128-NEXT: i32.add $push[[R:[0-9]+]]=, $1, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[R]]), $0{{$}}
define void @store_v2f64_with_unfolded_offset(<2 x double> %v, <2 x double>* %p) {
  %s = getelementptr inbounds <2 x double>, <2 x double>* %p, i32 -1
  store <2 x double> %v , <2 x double>* %s
  ret void
}

; CHECK-LABEL: store_v2f64_with_unfolded_gep_offset:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype store_v2f64_with_unfolded_gep_offset (v128, i32) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 16{{$}}
; SIMD128-NEXT: i32.add $push[[R:[0-9]+]]=, $1, $pop[[L0]]{{$}}
; SIMD128-NEXT: v128.store 0($pop[[R]]), $0{{$}}
define void @store_v2f64_with_unfolded_gep_offset(<2 x double> %v, <2 x double>* %p) {
  %s = getelementptr <2 x double>, <2 x double>* %p, i32 1
  store <2 x double> %v , <2 x double>* %s
  ret void
}

; CHECK-LABEL: store_v2f64_to_numeric_address:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype store_v2f64_to_numeric_address (v128) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.store 32($pop[[L0]]), $0{{$}}
define void @store_v2f64_to_numeric_address(<2 x double> %v) {
  %s = inttoptr i32 32 to <2 x double>*
  store <2 x double> %v , <2 x double>* %s
  ret void
}

; CHECK-LABEL: store_v2f64_to_global_address:
; NO-SIMD128-NOT: v128
; SIMD128-VM-NOT: v128
; SIMD128-NEXT: .functype store_v2f64_to_global_address (v128) -> (){{$}}
; SIMD128-NEXT: i32.const $push[[R:[0-9]+]]=, 0{{$}}
; SIMD128-NEXT: v128.store gv_v2f64($pop[[R]]), $0{{$}}
define void @store_v2f64_to_global_address(<2 x double> %v) {
  store <2 x double> %v , <2 x double>* @gv_v2f64
  ret void
}
