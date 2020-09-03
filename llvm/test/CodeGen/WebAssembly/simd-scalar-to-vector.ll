; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+simd128 | FileCheck %s

; Test that scalar_to_vector is lowered into a splat correctly.
; This bugpoint-reduced code turns into the selection dag below.
; TODO: find small test cases that produce scalar_to_vector dag nodes
; to make this test more readable and comprehensive.

;   t0: ch = EntryToken
; t32: i32,ch = load<(load 4 from `<2 x i16>* undef`, align 1)> t0, undef:i32, undef:i32
;   t33: v4i32 = scalar_to_vector t32
; t34: v8i16 = bitcast t33
;       t51: i32 = extract_vector_elt t34, Constant:i32<0>
;   t52: ch = store<(store 2 into `<4 x i16>* undef`, align 1), trunc to i16> t32:1, t51, undef:i32, undef:i32
;       t50: i32 = extract_vector_elt t34, Constant:i32<1>
;     t53: ch = store<(store 2 into `<4 x i16>* undef` + 2, align 1), trunc to i16> t32:1, t50, undef:i32, undef:i32
;       t49: i32 = extract_vector_elt t34, Constant:i32<2>
;     t55: ch = store<(store 2 into `<4 x i16>* undef` + 4, align 1), trunc to i16> t32:1, t49, undef:i32, undef:i32
;       t48: i32 = extract_vector_elt t34, Constant:i32<3>
;     t57: ch = store<(store 2 into `<4 x i16>* undef` + 6, align 1), trunc to i16> t32:1, t48, undef:i32, undef:i32
;   t58: ch = TokenFactor t52, t53, t55, t57
; t24: ch = WebAssemblyISD::RETURN t58

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: foo:
; CHECK: i64x2.splat
define void @foo() {
entry:
  %a = load <2 x i16>, <2 x i16>* undef, align 1
  %b = shufflevector <2 x i16> %a, <2 x i16> undef, <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %0 = bitcast <8 x i16> %b to <16 x i8>
  %shuffle.i214 = shufflevector <16 x i8> %0, <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef>, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  %1 = bitcast <16 x i8> %shuffle.i214 to <8 x i16>
  %add82 = add <8 x i16> %1, zeroinitializer
  %2 = select <8 x i1> undef, <8 x i16> undef, <8 x i16> %add82
  %3 = bitcast <8 x i16> %2 to <16 x i8>
  %shuffle.i204 = shufflevector <16 x i8> %3, <16 x i8> undef, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %4 = bitcast <16 x i8> %shuffle.i204 to <8 x i16>
  %dst2.0.vec.extract = shufflevector <8 x i16> %4, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x i16> %dst2.0.vec.extract, <4 x i16>* undef, align 1
  ret void
}
