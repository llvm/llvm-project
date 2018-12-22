; RUN: llc < %s -mattr=+simd128,+sign-ext -verify-machineinstrs

;; Ensures fastisel produces valid code when storing and loading split
;; up v2i64 values. This is a regression test for a bug that crashed
;; llc after fastisel produced machineinstrs that used registers that
;; had never been defined.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

define i64 @foo(<2 x i64> %vec) #0 {
entry:
  %vec.addr = alloca <2 x i64>, align 16
  store <2 x i64> %vec, <2 x i64>* %vec.addr, align 16
  %0 = load <2 x i64>, <2 x i64>* %vec.addr, align 16
  %1 = extractelement <2 x i64> %0, i32 0
  ret i64 %1
}

attributes #0 = { noinline optnone }
