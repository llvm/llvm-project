; RUN: opt -mtriple=x86_64-unknown-linux-gnu -load-store-vectorizer -S -o - %s | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; Vectorized subsets of the load/store chains in the presence of
; interleaved loads/stores

; CHECK-LABEL: @interleave_2L_2S(
; CHECK: load <2 x i32>
; CHECK: load i32
; CHECK: store <2 x i32>
; CHECK: load i32
define void @interleave_2L_2S(i32* noalias %ptr) {
  %next.gep = getelementptr i32, i32* %ptr, i64 0
  %next.gep1 = getelementptr i32, i32* %ptr, i64 1
  %next.gep2 = getelementptr i32, i32* %ptr, i64 2

  %l1 = load i32, i32* %next.gep1, align 4
  %l2 = load i32, i32* %next.gep, align 4
  store i32 0, i32* %next.gep1, align 4
  store i32 0, i32* %next.gep, align 4
  %l3 = load i32, i32* %next.gep1, align 4
  %l4 = load i32, i32* %next.gep2, align 4

  ret void
}

; CHECK-LABEL: @interleave_3L_2S_1L(
; CHECK: load <3 x i32>
; CHECK: store <2 x i32>
; CHECK: load i32

define void @interleave_3L_2S_1L(i32* noalias %ptr) {
  %next.gep = getelementptr i32, i32* %ptr, i64 0
  %next.gep1 = getelementptr i32, i32* %ptr, i64 1
  %next.gep2 = getelementptr i32, i32* %ptr, i64 2

  %l2 = load i32, i32* %next.gep, align 4
  %l1 = load i32, i32* %next.gep1, align 4
  store i32 0, i32* %next.gep1, align 4
  store i32 0, i32* %next.gep, align 4
  %l3 = load i32, i32* %next.gep1, align 4
  %l4 = load i32, i32* %next.gep2, align 4

  ret void
}

; CHECK-LABEL: @chain_suffix(
; CHECK: load i32
; CHECK: store <2 x i32>
; CHECK: load <2 x i32>
define void @chain_suffix(i32* noalias %ptr) {
  %next.gep = getelementptr i32, i32* %ptr, i64 0
  %next.gep1 = getelementptr i32, i32* %ptr, i64 1
  %next.gep2 = getelementptr i32, i32* %ptr, i64 2

  %l2 = load i32, i32* %next.gep, align 4
  store i32 0, i32* %next.gep1, align 4
  store i32 0, i32* %next.gep, align 4
  %l3 = load i32, i32* %next.gep1, align 4
  %l4 = load i32, i32* %next.gep2, align 4

  ret void
}


; CHECK-LABEL: @chain_prefix_suffix(
; CHECK: load <2 x i32>
; CHECK: store <2 x i32>
; CHECK: load <3 x i32>
define void  @chain_prefix_suffix(i32* noalias %ptr) {
  %next.gep = getelementptr i32, i32* %ptr, i64 0
  %next.gep1 = getelementptr i32, i32* %ptr, i64 1
  %next.gep2 = getelementptr i32, i32* %ptr, i64 2
  %next.gep3 = getelementptr i32, i32* %ptr, i64 3

  %l1 = load i32, i32* %next.gep, align 4
  %l2 = load i32, i32* %next.gep1, align 4
  store i32 0, i32* %next.gep1, align 4
  store i32 0, i32* %next.gep2, align 4
  %l3 = load i32, i32* %next.gep1, align 4
  %l4 = load i32, i32* %next.gep2, align 4
  %l5 = load i32, i32* %next.gep3, align 4

  ret void
}

