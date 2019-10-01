; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+unimplemented-simd128 | FileCheck %s

; Check that store in memory with smaller lanes are loaded and stored
; as expected. This is a regression test for part of bug 39275.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: load_ext_2xi32:
; CHECK-NEXT: .functype load_ext_2xi32 (i32) -> (v128){{$}}
; CHECK-NEXT: i64.load32_u $push[[L0:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: i64x2.splat $push[[L1:[0-9]+]]=, $pop[[L0]]{{$}}
; CHECK-NEXT: i64.load32_u $push[[L2:[0-9]+]]=, 4($0){{$}}
; CHECK-NEXT: i64x2.replace_lane $push[[R:[0-9]+]]=, $pop[[L1]], 1, $pop[[L2]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x i32> @load_ext_2xi32(<2 x i32>* %p) {
  %1 = load <2 x i32>, <2 x i32>* %p, align 4
  ret <2 x i32> %1
}

; CHECK-LABEL: load_zext_2xi32:
; CHECK-NEXT: .functype load_zext_2xi32 (i32) -> (v128){{$}}
; CHECK-NEXT: i64.load32_u $push[[L0:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: i64x2.splat $push[[L1:[0-9]+]]=, $pop[[L0]]{{$}}
; CHECK-NEXT: i64.load32_u $push[[L2:[0-9]+]]=, 4($0){{$}}
; CHECK-NEXT: i64x2.replace_lane $push[[R:[0-9]+]]=, $pop[[L1]], 1, $pop[[L2]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_zext_2xi32(<2 x i32>* %p) {
  %1 = load <2 x i32>, <2 x i32>* %p, align 4
  %2 = zext <2 x i32> %1 to <2 x i64>
  ret <2 x i64> %2
}

; CHECK-LABEL: load_sext_2xi32:
; CHECK-NEXT: .functype load_sext_2xi32 (i32) -> (v128){{$}}
; CHECK-NEXT: i64.load32_s $push[[L0:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: i64x2.splat $push[[L1:[0-9]+]]=, $pop[[L0]]{{$}}
; CHECK-NEXT: i64.load32_s $push[[L2:[0-9]+]]=, 4($0){{$}}
; CHECK-NEXT: i64x2.replace_lane $push[[R:[0-9]+]]=, $pop[[L1]], 1, $pop[[L2]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @load_sext_2xi32(<2 x i32>* %p) {
  %1 = load <2 x i32>, <2 x i32>* %p, align 4
  %2 = sext <2 x i32> %1 to <2 x i64>
  ret <2 x i64> %2
}

; CHECK-LABEL: store_trunc_2xi32:
; CHECK-NEXT: .functype store_trunc_2xi32 (i32, v128) -> (){{$}}
; CHECK-NEXT: i64x2.extract_lane $push[[L0:[0-9]+]]=, $1, 1
; CHECK-NEXT: i64.store32 4($0), $pop[[L0]]
; CHECK-NEXT: i64x2.extract_lane $push[[L1:[0-9]+]]=, $1, 0
; CHECK-NEXT: i64.store32 0($0), $pop[[L1]]
; CHECK-NEXT: return
define void @store_trunc_2xi32(<2 x i32>* %p, <2 x i32> %x) {
  store <2 x i32> %x, <2 x i32>* %p, align 4
  ret void
}
