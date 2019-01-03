; RUN: llc < %s -fast-isel -mattr=+simd128,+sign-ext -verify-machineinstrs

;; Ensures fastisel produces valid code when storing and loading split
;; up v2i64 values. Lowering away v2i64s is a temporary measure while
;; V8 does not have support for i64x2.* operations, and is done when
;; -wasm-enable-unimplemented-simd is not present. This is a
;; regression test for a bug that crashed llc after fastisel produced
;; machineinstrs that used registers that had never been defined.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define i64 @foo(<2 x i64> %vec) {
entry:
  %vec.addr = alloca <2 x i64>, align 16
  store <2 x i64> %vec, <2 x i64>* %vec.addr, align 16
  %0 = load <2 x i64>, <2 x i64>* %vec.addr, align 16
  %1 = extractelement <2 x i64> %0, i32 0
  ret i64 %1
}
