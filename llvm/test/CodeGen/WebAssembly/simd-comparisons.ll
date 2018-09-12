; RUN: llc < %s -disable-wasm-fallthrough-return-opt -wasm-keep-registers -wasm-disable-explicit-locals -wasm-enable-unimplemented-simd -mattr=+simd128,+sign-ext --show-mc-encoding | FileCheck %s --check-prefixes CHECK,SIMD128
; RUN: llc < %s -disable-wasm-fallthrough-return-opt -wasm-keep-registers -wasm-disable-explicit-locals -mattr=+simd128,+sign-ext --show-mc-encoding | FileCheck %s --check-prefixes CHECK,SIMD128-VM
; RUN: llc < %s -disable-wasm-fallthrough-return-opt -wasm-keep-registers -wasm-disable-explicit-locals -mattr=-simd128,+sign-ext --show-mc-encoding | FileCheck %s --check-prefixes CHECK,NO-SIMD128

; Test SIMD comparison operators

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: compare_eq_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i8x16.eq $push0=, $0, $1 # encoding: [0xfd,0x48]{{$}}
; SIMD128: return $pop0 #
define <16 x i1> @compare_eq_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp eq <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_ne_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i8x16.ne $push0=, $0, $1 # encoding: [0xfd,0x4d]{{$}}
; SIMD128: return $pop0 #
define <16 x i1> @compare_ne_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp ne <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_slt_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i8x16.lt_s $push0=, $0, $1 # encoding: [0xfd,0x52]{{$}}
; SIMD128: return $pop0 #
define <16 x i1> @compare_slt_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp slt <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_ult_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i8x16.lt_u $push0=, $0, $1 # encoding: [0xfd,0x53]{{$}}
; SIMD128: return $pop0 #
define <16 x i1> @compare_ult_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp ult <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_sle_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i8x16.le_s $push0=, $0, $1 # encoding: [0xfd,0x5a]{{$}}
; SIMD128: return $pop0 #
define <16 x i1> @compare_sle_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp sle <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_ule_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i8x16.le_u $push0=, $0, $1 # encoding: [0xfd,0x5b]{{$}}
; SIMD128: return $pop0 #
define <16 x i1> @compare_ule_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp ule <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_sgt_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i8x16.gt_s $push0=, $0, $1 # encoding: [0xfd,0x62]{{$}}
; SIMD128: return $pop0 #
define <16 x i1> @compare_sgt_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp sgt <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_ugt_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i8x16.gt_u $push0=, $0, $1 # encoding: [0xfd,0x63]{{$}}
; SIMD128: return $pop0 #
define <16 x i1> @compare_ugt_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp ugt <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_sge_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i8x16.ge_s $push0=, $0, $1 # encoding: [0xfd,0x6a]{{$}}
; SIMD128: return $pop0 #
define <16 x i1> @compare_sge_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp sge <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_uge_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i8x16.ge_u $push0=, $0, $1 # encoding: [0xfd,0x6b]{{$}}
; SIMD128: return $pop0 #
define <16 x i1> @compare_uge_v16i8 (<16 x i8> %x, <16 x i8> %y) {
  %res = icmp uge <16 x i8> %x, %y
  ret <16 x i1> %res
}

; CHECK-LABEL: compare_eq_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i16x8.eq $push0=, $0, $1 # encoding: [0xfd,0x49]{{$}}
; SIMD128: return $pop0 #
define <8 x i1> @compare_eq_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp eq <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_ne_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i16x8.ne $push0=, $0, $1 # encoding: [0xfd,0x4e]{{$}}
; SIMD128: return $pop0 #
define <8 x i1> @compare_ne_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp ne <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_slt_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i16x8.lt_s $push0=, $0, $1 # encoding: [0xfd,0x54]{{$}}
; SIMD128: return $pop0 #
define <8 x i1> @compare_slt_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp slt <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_ult_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i16x8.lt_u $push0=, $0, $1 # encoding: [0xfd,0x55]{{$}}
; SIMD128: return $pop0 #
define <8 x i1> @compare_ult_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp ult <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_sle_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i16x8.le_s $push0=, $0, $1 # encoding: [0xfd,0x5c]{{$}}
; SIMD128: return $pop0 #
define <8 x i1> @compare_sle_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp sle <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_ule_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i16x8.le_u $push0=, $0, $1 # encoding: [0xfd,0x5d]{{$}}
; SIMD128: return $pop0 #
define <8 x i1> @compare_ule_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp ule <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_sgt_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i16x8.gt_s $push0=, $0, $1 # encoding: [0xfd,0x64]{{$}}
; SIMD128: return $pop0 #
define <8 x i1> @compare_sgt_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp sgt <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_ugt_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i16x8.gt_u $push0=, $0, $1 # encoding: [0xfd,0x65]{{$}}
; SIMD128: return $pop0 #
define <8 x i1> @compare_ugt_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp ugt <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_sge_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i16x8.ge_s $push0=, $0, $1 # encoding: [0xfd,0x6c]{{$}}
; SIMD128: return $pop0 #
define <8 x i1> @compare_sge_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp sge <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_uge_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i16x8.ge_u $push0=, $0, $1 # encoding: [0xfd,0x6d]{{$}}
; SIMD128: return $pop0 #
define <8 x i1> @compare_uge_v8i16 (<8 x i16> %x, <8 x i16> %y) {
  %res = icmp uge <8 x i16> %x, %y
  ret <8 x i1> %res
}

; CHECK-LABEL: compare_eq_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i32x4.eq $push0=, $0, $1 # encoding: [0xfd,0x4a]{{$}}
; SIMD128: return $pop0 #
define <4 x i1> @compare_eq_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp eq <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_ne_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i32x4.ne $push0=, $0, $1 # encoding: [0xfd,0x4f]{{$}}
; SIMD128: return $pop0 #
define <4 x i1> @compare_ne_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp ne <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_slt_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i32x4.lt_s $push0=, $0, $1 # encoding: [0xfd,0x56]{{$}}
; SIMD128: return $pop0 #
define <4 x i1> @compare_slt_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp slt <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_ult_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i32x4.lt_u $push0=, $0, $1 # encoding: [0xfd,0x57]{{$}}
; SIMD128: return $pop0 #
define <4 x i1> @compare_ult_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp ult <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sle_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i32x4.le_s $push0=, $0, $1 # encoding: [0xfd,0x5e]{{$}}
; SIMD128: return $pop0 #
define <4 x i1> @compare_sle_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp sle <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_ule_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i32x4.le_u $push0=, $0, $1 # encoding: [0xfd,0x5f]{{$}}
; SIMD128: return $pop0 #
define <4 x i1> @compare_ule_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp ule <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sgt_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i32x4.gt_s $push0=, $0, $1 # encoding: [0xfd,0x66]{{$}}
; SIMD128: return $pop0 #
define <4 x i1> @compare_sgt_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp sgt <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_ugt_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i32x4.gt_u $push0=, $0, $1 # encoding: [0xfd,0x67]{{$}}
; SIMD128: return $pop0 #
define <4 x i1> @compare_ugt_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp ugt <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_sge_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i32x4.ge_s $push0=, $0, $1 # encoding: [0xfd,0x6e]{{$}}
; SIMD128: return $pop0 #
define <4 x i1> @compare_sge_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp sge <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_uge_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i32x4.ge_u $push0=, $0, $1 # encoding: [0xfd,0x6f]{{$}}
; SIMD128: return $pop0 #
define <4 x i1> @compare_uge_v4i32 (<4 x i32> %x, <4 x i32> %y) {
  %res = icmp uge <4 x i32> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_oeq_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f32x4.eq $push0=, $0, $1 # encoding: [0xfd,0x4b]{{$}}
; SIMD128: return $pop0 #
define <4 x i1> @compare_oeq_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp oeq <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_ogt_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f32x4.gt $push0=, $0, $1 # encoding: [0xfd,0x68]{{$}}
; SIMD128: return $pop0 #
define <4 x i1> @compare_ogt_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp ogt <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_oge_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f32x4.ge $push0=, $0, $1 # encoding: [0xfd,0x70]{{$}}
; SIMD128: return $pop0 #
define <4 x i1> @compare_oge_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp oge <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_olt_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f32x4.lt $push0=, $0, $1 # encoding: [0xfd,0x58]{{$}}
; SIMD128: return $pop0 #
define <4 x i1> @compare_olt_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp olt <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_ole_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f32x4.le $push0=, $0, $1 # encoding: [0xfd,0x60]{{$}}
; SIMD128: return $pop0 #
define <4 x i1> @compare_ole_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp ole <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_one_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f32x4.ne
define <4 x i1> @compare_one_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp one <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_ord_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f32x4.eq
define <4 x i1> @compare_ord_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp ord <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_ueq_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f32x4.eq
define <4 x i1> @compare_ueq_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp ueq <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_ugt_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f32x4.le
define <4 x i1> @compare_ugt_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp ugt <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_uge_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f32x4.lt
define <4 x i1> @compare_uge_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp uge <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_ult_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f32x4.ge
define <4 x i1> @compare_ult_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp ult <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_ule_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f32x4.gt
define <4 x i1> @compare_ule_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp ule <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_une_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f32x4.ne $push0=, $0, $1 # encoding: [0xfd,0x50]{{$}}
; SIMD128: return $pop0 #
define <4 x i1> @compare_une_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp une <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_uno_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f32x4.ne
define <4 x i1> @compare_uno_v4f32 (<4 x float> %x, <4 x float> %y) {
  %res = fcmp uno <4 x float> %x, %y
  ret <4 x i1> %res
}

; CHECK-LABEL: compare_oeq_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f64x2.eq $push0=, $0, $1 # encoding: [0xfd,0x4c]{{$}}
; SIMD128: return $pop0 #
define <2 x i1> @compare_oeq_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp oeq <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_ogt_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f64x2.gt $push0=, $0, $1 # encoding: [0xfd,0x69]{{$}}
; SIMD128: return $pop0 #
define <2 x i1> @compare_ogt_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp ogt <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_oge_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f64x2.ge $push0=, $0, $1 # encoding: [0xfd,0x71]{{$}}
; SIMD128: return $pop0 #
define <2 x i1> @compare_oge_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp oge <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_olt_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f64x2.lt $push0=, $0, $1 # encoding: [0xfd,0x59]{{$}}
; SIMD128: return $pop0 #
define <2 x i1> @compare_olt_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp olt <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_ole_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f64x2.le $push0=, $0, $1 # encoding: [0xfd,0x61]{{$}}
; SIMD128: return $pop0 #
define <2 x i1> @compare_ole_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp ole <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_one_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f64x2.ne
define <2 x i1> @compare_one_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp one <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_ord_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f64x2.eq
define <2 x i1> @compare_ord_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp ord <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_ueq_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f64x2.eq
define <2 x i1> @compare_ueq_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp ueq <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_ugt_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f64x2.le
define <2 x i1> @compare_ugt_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp ugt <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_uge_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f64x2.lt
define <2 x i1> @compare_uge_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp uge <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_ult_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f64x2.ge
define <2 x i1> @compare_ult_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp ult <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_ule_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f64x2.gt
define <2 x i1> @compare_ule_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp ule <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_une_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f64x2.ne $push0=, $0, $1 # encoding: [0xfd,0x51]{{$}}
; SIMD128: return $pop0 #
define <2 x i1> @compare_une_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp une <2 x double> %x, %y
  ret <2 x i1> %res
}

; CHECK-LABEL: compare_uno_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128: .param v128, v128{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f64x2.ne
define <2 x i1> @compare_uno_v2f64 (<2 x double> %x, <2 x double> %y) {
  %res = fcmp uno <2 x double> %x, %y
  ret <2 x i1> %res
}
