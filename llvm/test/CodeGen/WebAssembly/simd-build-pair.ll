; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=simd128 | FileCheck %s --check-prefixes CHECK

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; Test that BUILD_PAIR dag nodes are correctly lowered.
; This code produces a selection DAG like the following:

;    t0: ch = EntryToken
;  t3: v4i32,ch = load<(load 16 from `<4 x i32>* undef`)> t0, undef:i32, undef:i32
;        t30: i32 = extract_vector_elt t3, Constant:i32<2>
;        t28: i32 = extract_vector_elt t3, Constant:i32<3>
;      t24: i64 = build_pair t30, t28
;    t8: ch = store<(store 8 into `i64* undef`, align 1)> t3:1, t24, undef:i32, undef:i32
;  t9: ch = WebAssemblyISD::RETURN t8

; CHECK:      i64x2.extract_lane
; CHECK-NEXT: i64.store
define void @build_pair_i32s() {
entry:
  %0 = load <4 x i32>, <4 x i32>* undef, align 16
  %shuffle.i184 = shufflevector <4 x i32> %0, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bc357 = bitcast <4 x i32> %shuffle.i184 to <2 x i64>
  %1 = extractelement <2 x i64> %bc357, i32 0
  store i64 %1, i64* undef, align 1
  ret void
}
