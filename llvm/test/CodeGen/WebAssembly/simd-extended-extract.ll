; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -mattr=+simd128 | FileCheck %s

; Regression test for an issue with patterns like the following:
;
;     t101: v4i32 = BUILD_VECTOR t99, t99, t99, t99
;         t92: i32 = extract_vector_elt t101, Constant:i32<0>
;             t89: i32 = sign_extend_inreg t92, ValueType:ch:i8
;
; Notice that the sign_extend_inreg has source value type i8 but the
; extracted vector has type v4i32. There are no ISel patterns that
; handle mismatched types like this, so we insert a bitcast before the
; extract. This was previously an ISel failure. This test case is
; reduced from a private user bug report, and the vector extracts are
; optimized out via subsequent DAG combines.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: foo:
; CHECK:         .functype foo (i32) -> ()
; Implementation omitted...
; CHECK:         return
define void @foo(<4 x i8>* %p) {
  %1 = load <4 x i8>, <4 x i8>* %p
  %2 = sitofp <4 x i8> %1 to <4 x double>
  %3 = fmul <4 x double> zeroinitializer, %2
  %4 = fadd <4 x double> %3, zeroinitializer
  %5 = fptrunc <4 x double> %4 to <4 x float>
  store <4 x float> %5, <4 x float>* undef
  ret void
}
