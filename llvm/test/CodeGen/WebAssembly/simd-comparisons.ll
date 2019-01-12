; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-keep-registers -wasm-disable-explicit-locals -mattr=+unimplemented-simd128 | FileCheck %s --check-prefixes CHECK,SIMD128
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-keep-registers -wasm-disable-explicit-locals -mattr=+simd128 | FileCheck %s --check-prefixes CHECK,SIMD128-VM
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-keep-registers -wasm-disable-explicit-locals | FileCheck %s --check-prefixes CHECK,NO-SIMD128

; Test SIMD comparison operators

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: compare_eq_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_eq_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.eq $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i1> @compare_eq_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp eq <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_sext_eq_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_sext_eq_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.eq $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @compare_sext_eq_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %cmp = icmp eq <16 x i8> %x, %y
  %res = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %res
}

; CHECK-LABEL: compare_ne_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_ne_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.ne $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i1> @compare_ne_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp ne <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_sext_ne_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_sext_ne_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.ne $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @compare_sext_ne_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %cmp = icmp ne <16 x i8> %x, %y
  %res = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %res
}

; CHECK-LABEL: compare_slt_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_slt_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.lt_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i1> @compare_slt_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp slt <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_sext_slt_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_sext_slt_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.lt_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @compare_sext_slt_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %cmp = icmp slt <16 x i8> %x, %y
  %res = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %res
}

; CHECK-LABEL: compare_ult_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_ult_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.lt_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i1> @compare_ult_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp ult <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_sext_ult_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_sext_ult_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.lt_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @compare_sext_ult_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %cmp = icmp ult <16 x i8> %x, %y
  %res = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %res
}

; CHECK-LABEL: compare_sle_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_sle_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.le_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i1> @compare_sle_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp sle <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_sext_sle_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_sext_sle_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.le_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @compare_sext_sle_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %cmp = icmp sle <16 x i8> %x, %y
  %res = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %res
}

; CHECK-LABEL: compare_ule_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_ule_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.le_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i1> @compare_ule_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp ule <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_sext_ule_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_sext_ule_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.le_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @compare_sext_ule_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %cmp = icmp ule <16 x i8> %x, %y
  %res = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %res
}

; CHECK-LABEL: compare_sgt_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_sgt_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.gt_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i1> @compare_sgt_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp sgt <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_sext_sgt_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_sext_sgt_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.gt_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @compare_sext_sgt_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %cmp = icmp sgt <16 x i8> %x, %y
  %res = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %res
}

; CHECK-LABEL: compare_ugt_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_ugt_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.gt_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i1> @compare_ugt_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp ugt <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_sext_ugt_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_sext_ugt_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.gt_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @compare_sext_ugt_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %cmp = icmp ugt <16 x i8> %x, %y
  %res = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %res
}

; CHECK-LABEL: compare_sge_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_sge_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.ge_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i1> @compare_sge_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp sge <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_sext_sge_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_sext_sge_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.ge_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @compare_sext_sge_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %cmp = icmp sge <16 x i8> %x, %y
  %res = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %res
}

; CHECK-LABEL: compare_uge_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_uge_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.ge_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i1> @compare_uge_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp uge <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_sext_uge_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype compare_sext_uge_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.ge_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @compare_sext_uge_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %cmp = icmp uge <16 x i8> %x, %y
  %res = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %res
}

; CHECK-LABEL: compare_eq_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_eq_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.eq $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i1> @compare_eq_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp eq <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_sext_eq_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_sext_eq_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.eq $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @compare_sext_eq_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %cmp = icmp eq <8 x i16> %x, %y
  %res = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %res
}

; CHECK-LABEL: compare_ne_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_ne_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.ne $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i1> @compare_ne_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp ne <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_sext_ne_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_sext_ne_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.ne $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @compare_sext_ne_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %cmp = icmp ne <8 x i16> %x, %y
  %res = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %res
}

; CHECK-LABEL: compare_slt_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_slt_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.lt_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i1> @compare_slt_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp slt <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_sext_slt_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_sext_slt_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.lt_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @compare_sext_slt_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %cmp = icmp slt <8 x i16> %x, %y
  %res = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %res
}

; CHECK-LABEL: compare_ult_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_ult_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.lt_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i1> @compare_ult_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp ult <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_sext_ult_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_sext_ult_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.lt_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @compare_sext_ult_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %cmp = icmp ult <8 x i16> %x, %y
  %res = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %res
}

; CHECK-LABEL: compare_sle_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_sle_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.le_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i1> @compare_sle_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp sle <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_sext_sle_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_sext_sle_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.le_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @compare_sext_sle_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %cmp = icmp sle <8 x i16> %x, %y
  %res = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %res
}

; CHECK-LABEL: compare_ule_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_ule_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.le_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i1> @compare_ule_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp ule <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_sext_ule_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_sext_ule_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.le_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @compare_sext_ule_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %cmp = icmp ule <8 x i16> %x, %y
  %res = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %res
}

; CHECK-LABEL: compare_sgt_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_sgt_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.gt_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i1> @compare_sgt_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp sgt <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_sext_sgt_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_sext_sgt_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.gt_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @compare_sext_sgt_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %cmp = icmp sgt <8 x i16> %x, %y
  %res = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %res
}

; CHECK-LABEL: compare_ugt_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_ugt_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.gt_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i1> @compare_ugt_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp ugt <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_sext_ugt_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_sext_ugt_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.gt_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @compare_sext_ugt_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %cmp = icmp ugt <8 x i16> %x, %y
  %res = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %res
}

; CHECK-LABEL: compare_sge_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_sge_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.ge_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i1> @compare_sge_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp sge <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_sext_sge_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_sext_sge_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.ge_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @compare_sext_sge_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %cmp = icmp sge <8 x i16> %x, %y
  %res = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %res
}

; CHECK-LABEL: compare_uge_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_uge_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.ge_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i1> @compare_uge_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp uge <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_sext_uge_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype compare_sext_uge_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.ge_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @compare_sext_uge_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %cmp = icmp uge <8 x i16> %x, %y
  %res = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %res
}

; CHECK-LABEL: compare_eq_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_eq_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.eq $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_eq_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp eq <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_eq_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_sext_eq_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.eq $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_eq_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %cmp = icmp eq <4 x i32> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_ne_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_ne_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.ne $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_ne_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp ne <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_ne_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_sext_ne_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.ne $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_ne_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %cmp = icmp ne <4 x i32> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_slt_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_slt_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.lt_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_slt_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp slt <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_slt_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_sext_slt_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.lt_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_slt_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %cmp = icmp slt <4 x i32> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_ult_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_ult_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.lt_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_ult_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp ult <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_ult_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_sext_ult_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.lt_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_ult_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %cmp = icmp ult <4 x i32> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_sle_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_sle_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.le_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_sle_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp sle <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_sle_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_sext_sle_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.le_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_sle_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %cmp = icmp sle <4 x i32> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_ule_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_ule_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.le_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_ule_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp ule <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_ule_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_sext_ule_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.le_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_ule_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %cmp = icmp ule <4 x i32> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_sgt_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_sgt_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.gt_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_sgt_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp sgt <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_sgt_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_sext_sgt_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.gt_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_sgt_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %cmp = icmp sgt <4 x i32> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_ugt_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_ugt_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.gt_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_ugt_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp ugt <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_ugt_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_sext_ugt_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.gt_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_ugt_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %cmp = icmp ugt <4 x i32> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_sge_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_sge_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.ge_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_sge_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp sge <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_sge_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_sext_sge_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.ge_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_sge_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %cmp = icmp sge <4 x i32> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_uge_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_uge_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.ge_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_uge_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp uge <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_uge_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype compare_sext_uge_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.ge_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_uge_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %cmp = icmp uge <4 x i32> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_oeq_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_oeq_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.eq $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_oeq_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp oeq <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_oeq_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_sext_oeq_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.eq $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_oeq_v4f32 (<4 x float> %x, <4 x float> %y) {
  %cmp = fcmp oeq <4 x float> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_ogt_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_ogt_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.gt $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_ogt_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp ogt <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_ogt_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_sext_ogt_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.gt $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_ogt_v4f32 (<4 x float> %x, <4 x float> %y) {
  %cmp = fcmp ogt <4 x float> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_oge_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_oge_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.ge $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_oge_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp oge <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_oge_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_sext_oge_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.ge $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_oge_v4f32 (<4 x float> %x, <4 x float> %y) {
  %cmp = fcmp oge <4 x float> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_olt_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_olt_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.lt $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_olt_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp olt <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_olt_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_sext_olt_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.lt $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_olt_v4f32 (<4 x float> %x, <4 x float> %y) {
  %cmp = fcmp olt <4 x float> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_ole_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_ole_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.le $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_ole_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp ole <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_ole_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_sext_ole_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.le $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_ole_v4f32 (<4 x float> %x, <4 x float> %y) {
  %cmp = fcmp ole <4 x float> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_one_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_one_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.ne $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: f32x4.eq $push[[T1:[0-9]+]]=, $0, $0{{$}}
; SIMD128-NEXT: f32x4.eq $push[[T2:[0-9]+]]=, $1, $1{{$}}
; SIMD128-NEXT: v128.and $push[[T3:[0-9]+]]=, $pop[[T1]], $pop[[T2]]{{$}}
; SIMD128-NEXT: v128.and $push[[R:[0-9]+]]=, $pop[[T0]], $pop[[T3]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_one_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp one <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_one_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_sext_one_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.ne $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: f32x4.eq $push[[T1:[0-9]+]]=, $0, $0{{$}}
; SIMD128-NEXT: f32x4.eq $push[[T2:[0-9]+]]=, $1, $1{{$}}
; SIMD128-NEXT: v128.and $push[[T3:[0-9]+]]=, $pop[[T1]], $pop[[T2]]{{$}}
; SIMD128-NEXT: v128.and $push[[R:[0-9]+]]=, $pop[[T0]], $pop[[T3]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_one_v4f32 (<4 x float> %x, <4 x float> %y) {
  %cmp = fcmp one <4 x float> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_ord_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_ord_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.eq $push[[T0:[0-9]+]]=, $0, $0{{$}}
; SIMD128-NEXT: f32x4.eq $push[[T1:[0-9]+]]=, $1, $1{{$}}
; SIMD128-NEXT: v128.and $push[[R:[0-9]+]]=, $pop[[T0]], $pop[[T1]]{{$}}
; SIMD128-NEXT: return   $pop[[R]]{{$}}
define <4 x i1> @compare_ord_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp ord <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_ord_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_sext_ord_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.eq $push[[T0:[0-9]+]]=, $0, $0{{$}}
; SIMD128-NEXT: f32x4.eq $push[[T1:[0-9]+]]=, $1, $1{{$}}
; SIMD128-NEXT: v128.and $push[[R:[0-9]+]]=, $pop[[T0]], $pop[[T1]]{{$}}
; SIMD128-NEXT: return   $pop[[R]]{{$}}
define <4 x i32> @compare_sext_ord_v4f32 (<4 x float> %x, <4 x float> %y) {
  %cmp = fcmp ord <4 x float> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_ueq_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_ueq_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.eq $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: f32x4.ne $push[[T1:[0-9]+]]=, $0, $0{{$}}
; SIMD128-NEXT: f32x4.ne $push[[T2:[0-9]+]]=, $1, $1{{$}}
; SIMD128-NEXT: v128.or $push[[T3:[0-9]+]]=, $pop[[T1]], $pop[[T2]]{{$}}
; SIMD128-NEXT: v128.or $push[[R:[0-9]+]]=, $pop[[T0]], $pop[[T3]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_ueq_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp ueq <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_ueq_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_sext_ueq_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.eq $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: f32x4.ne $push[[T1:[0-9]+]]=, $0, $0{{$}}
; SIMD128-NEXT: f32x4.ne $push[[T2:[0-9]+]]=, $1, $1{{$}}
; SIMD128-NEXT: v128.or $push[[T3:[0-9]+]]=, $pop[[T1]], $pop[[T2]]{{$}}
; SIMD128-NEXT: v128.or $push[[R:[0-9]+]]=, $pop[[T0]], $pop[[T3]]{{$}}
; SIMD128-NEXT: return $pop[[R]]
define <4 x i32> @compare_sext_ueq_v4f32 (<4 x float> %x, <4 x float> %y) {
  %cmp = fcmp ueq <4 x float> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_ugt_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_ugt_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.le $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $pop[[T0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_ugt_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp ugt <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_ugt_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_sext_ugt_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.le $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $pop[[T0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_ugt_v4f32 (<4 x float> %x, <4 x float> %y) {
  %cmp = fcmp ugt <4 x float> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_uge_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_uge_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.lt $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $pop[[T0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_uge_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp uge <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_uge_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_sext_uge_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.lt $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $pop[[T0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_uge_v4f32 (<4 x float> %x, <4 x float> %y) {
  %cmp = fcmp uge <4 x float> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_ult_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_ult_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.ge $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $pop[[T0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_ult_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp ult <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_ult_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_sext_ult_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.ge $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $pop[[T0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_ult_v4f32 (<4 x float> %x, <4 x float> %y) {
  %cmp = fcmp ult <4 x float> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_ule_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_ule_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.gt $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $pop[[T0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_ule_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp ule <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_ule_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_sext_ule_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.gt $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $pop[[T0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_ule_v4f32 (<4 x float> %x, <4 x float> %y) {
  %cmp = fcmp ule <4 x float> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_une_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_une_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.ne $push[[R:[0-9]+]]=, $0, $1{{$}}{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_une_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp une <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_une_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_sext_une_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.ne $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_une_v4f32 (<4 x float> %x, <4 x float> %y) {
  %cmp = fcmp une <4 x float> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_uno_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_uno_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.ne $push[[T0:[0-9]+]]=, $0, $0{{$}}
; SIMD128-NEXT: f32x4.ne $push[[T1:[0-9]+]]=, $1, $1{{$}}
; SIMD128-NEXT: v128.or $push[[R:[0-9]+]]=, $pop[[T0]], $pop[[T1]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i1> @compare_uno_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp uno <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sext_uno_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype compare_sext_uno_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.ne $push[[T0:[0-9]+]]=, $0, $0{{$}}
; SIMD128-NEXT: f32x4.ne $push[[T1:[0-9]+]]=, $1, $1{{$}}
; SIMD128-NEXT: v128.or $push[[R:[0-9]+]]=, $pop[[T0]], $pop[[T1]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @compare_sext_uno_v4f32 (<4 x float> %x, <4 x float> %y) {
  %cmp = fcmp uno <4 x float> %x, %y
  %res = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: compare_oeq_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_oeq_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.eq $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i1> @compare_oeq_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp oeq <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_sext_oeq_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_sext_oeq_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.eq $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @compare_sext_oeq_v2f64 (<2 x double> %x, <2 x double> %y) {
  %cmp = fcmp oeq <2 x double> %x, %y
  %res = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: compare_ogt_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_ogt_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.gt $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i1> @compare_ogt_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp ogt <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_sext_ogt_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_sext_ogt_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.gt $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @compare_sext_ogt_v2f64 (<2 x double> %x, <2 x double> %y) {
  %cmp = fcmp ogt <2 x double> %x, %y
  %res = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: compare_oge_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_oge_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.ge $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i1> @compare_oge_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp oge <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_sext_oge_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_sext_oge_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.ge $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @compare_sext_oge_v2f64 (<2 x double> %x, <2 x double> %y) {
  %cmp = fcmp oge <2 x double> %x, %y
  %res = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: compare_olt_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_olt_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.lt $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i1> @compare_olt_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp olt <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_sext_olt_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_sext_olt_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.lt $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @compare_sext_olt_v2f64 (<2 x double> %x, <2 x double> %y) {
  %cmp = fcmp olt <2 x double> %x, %y
  %res = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: compare_ole_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_ole_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.le $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i1> @compare_ole_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp ole <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_sext_ole_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_sext_ole_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.le $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @compare_sext_ole_v2f64 (<2 x double> %x, <2 x double> %y) {
  %cmp = fcmp ole <2 x double> %x, %y
  %res = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: compare_one_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_one_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.ne $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: f64x2.eq $push[[T1:[0-9]+]]=, $0, $0{{$}}
; SIMD128-NEXT: f64x2.eq $push[[T2:[0-9]+]]=, $1, $1{{$}}
; SIMD128-NEXT: v128.and $push[[T3:[0-9]+]]=, $pop[[T1]], $pop[[T2]]{{$}}
; SIMD128-NEXT: v128.and $push[[R:[0-9]+]]=, $pop[[T0]], $pop[[T3]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i1> @compare_one_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp one <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_sext_one_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_sext_one_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.ne $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: f64x2.eq $push[[T1:[0-9]+]]=, $0, $0{{$}}
; SIMD128-NEXT: f64x2.eq $push[[T2:[0-9]+]]=, $1, $1{{$}}
; SIMD128-NEXT: v128.and $push[[T3:[0-9]+]]=, $pop[[T1]], $pop[[T2]]{{$}}
; SIMD128-NEXT: v128.and $push[[R:[0-9]+]]=, $pop[[T0]], $pop[[T3]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @compare_sext_one_v2f64 (<2 x double> %x, <2 x double> %y) {
  %cmp = fcmp one <2 x double> %x, %y
  %res = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: compare_ord_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_ord_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.eq $push[[T0:[0-9]+]]=, $0, $0{{$}}
; SIMD128-NEXT: f64x2.eq $push[[T1:[0-9]+]]=, $1, $1{{$}}
; SIMD128-NEXT: v128.and $push[[R:[0-9]+]]=, $pop[[T0]], $pop[[T1]]{{$}}
; SIMD128-NEXT: return   $pop[[R]]{{$}}
define <2 x i1> @compare_ord_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp ord <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_sext_ord_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_sext_ord_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.eq $push[[T0:[0-9]+]]=, $0, $0{{$}}
; SIMD128-NEXT: f64x2.eq $push[[T1:[0-9]+]]=, $1, $1{{$}}
; SIMD128-NEXT: v128.and $push[[R:[0-9]+]]=, $pop[[T0]], $pop[[T1]]{{$}}
; SIMD128-NEXT: return   $pop[[R]]{{$}}
define <2 x i64> @compare_sext_ord_v2f64 (<2 x double> %x, <2 x double> %y) {
  %cmp = fcmp ord <2 x double> %x, %y
  %res = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: compare_ueq_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_ueq_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.eq $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: f64x2.ne $push[[T1:[0-9]+]]=, $0, $0{{$}}
; SIMD128-NEXT: f64x2.ne $push[[T2:[0-9]+]]=, $1, $1{{$}}
; SIMD128-NEXT: v128.or $push[[T3:[0-9]+]]=, $pop[[T1]], $pop[[T2]]{{$}}
; SIMD128-NEXT: v128.or $push[[R:[0-9]+]]=, $pop[[T0]], $pop[[T3]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i1> @compare_ueq_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp ueq <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_sext_ueq_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_sext_ueq_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.eq $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: f64x2.ne $push[[T1:[0-9]+]]=, $0, $0{{$}}
; SIMD128-NEXT: f64x2.ne $push[[T2:[0-9]+]]=, $1, $1{{$}}
; SIMD128-NEXT: v128.or $push[[T3:[0-9]+]]=, $pop[[T1]], $pop[[T2]]{{$}}
; SIMD128-NEXT: v128.or $push[[R:[0-9]+]]=, $pop[[T0]], $pop[[T3]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @compare_sext_ueq_v2f64 (<2 x double> %x, <2 x double> %y) {
  %cmp = fcmp ueq <2 x double> %x, %y
  %res = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: compare_ugt_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_ugt_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.le $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $pop[[T0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i1> @compare_ugt_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp ugt <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_sext_ugt_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_sext_ugt_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.le $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $pop[[T0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @compare_sext_ugt_v2f64 (<2 x double> %x, <2 x double> %y) {
  %cmp = fcmp ugt <2 x double> %x, %y
  %res = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: compare_uge_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_uge_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.lt $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $pop[[T0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i1> @compare_uge_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp uge <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_sext_uge_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_sext_uge_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.lt $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $pop[[T0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @compare_sext_uge_v2f64 (<2 x double> %x, <2 x double> %y) {
  %cmp = fcmp uge <2 x double> %x, %y
  %res = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: compare_ult_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_ult_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.ge $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $pop[[T0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i1> @compare_ult_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp ult <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_sext_ult_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_sext_ult_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.ge $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $pop[[T0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @compare_sext_ult_v2f64 (<2 x double> %x, <2 x double> %y) {
  %cmp = fcmp ult <2 x double> %x, %y
  %res = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: compare_ule_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_ule_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.gt $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $pop[[T0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i1> @compare_ule_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp ule <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_sext_ule_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_sext_ule_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.gt $push[[T0:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $pop[[T0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @compare_sext_ule_v2f64 (<2 x double> %x, <2 x double> %y) {
  %cmp = fcmp ule <2 x double> %x, %y
  %res = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: compare_une_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_une_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.ne $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i1> @compare_une_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp une <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_sext_une_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_sext_une_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.ne $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @compare_sext_une_v2f64 (<2 x double> %x, <2 x double> %y) {
  %cmp = fcmp une <2 x double> %x, %y
  %res = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: compare_uno_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_uno_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.ne $push[[T0:[0-9]+]]=, $0, $0{{$}}
; SIMD128-NEXT: f64x2.ne $push[[T1:[0-9]+]]=, $1, $1{{$}}
; SIMD128-NEXT: v128.or $push[[R:[0-9]+]]=, $pop[[T0]], $pop[[T1]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i1> @compare_uno_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp uno <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_sext_uno_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128-NEXT: .functype compare_sext_uno_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.ne $push[[T0:[0-9]+]]=, $0, $0{{$}}
; SIMD128-NEXT: f64x2.ne $push[[T1:[0-9]+]]=, $1, $1{{$}}
; SIMD128-NEXT: v128.or $push[[R:[0-9]+]]=, $pop[[T0]], $pop[[T1]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @compare_sext_uno_v2f64 (<2 x double> %x, <2 x double> %y) {
  %cmp = fcmp uno <2 x double> %x, %y
  %res = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %res
}
