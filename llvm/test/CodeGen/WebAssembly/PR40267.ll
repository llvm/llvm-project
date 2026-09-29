; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers

; Regression test for PR40267. Tests that invalid indices in
; extract_vector_elt can be handled when vectors ops are split. Notice
; that SIMD is not enabled for this test. Check only that llc does not
; crash, since it would previously trigger an assertion.

target triple = "wasm32-unknown-unknown"

define void @foo() {
  %L6 = load i32, ptr undef
  br label %BB1

BB1:                                              ; preds = %BB1, %0
  %bj = select <4 x i1> <i1 true, i1 true, i1 false, i1 false>, <4 x i32> <i32 55, i32 21, i32 92, i32 68>, <4 x i32> <i32 51, i32 61, i32 62, i32 39>
  %E1 = extractelement <4 x i32> %bj, i32 0
  %E23 = extractelement <4 x i32> zeroinitializer, i32 %E1
  %I33 = insertelement <4 x i32> undef, i32 %E23, i1 undef
  store <4 x i32> %I33, ptr undef
  br label %BB1
}
