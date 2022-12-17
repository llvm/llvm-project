; RUN: opt -mtriple=x86_64-unknown-linux-gnu -passes=load-store-vectorizer -S -o - %s | FileCheck %s
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -aa-pipeline=basic-aa -passes='function(load-store-vectorizer)' -S -o - %s | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: @correct_order(
; CHECK: [[LOAD_PTR:%[0-9]+]] = bitcast i32* %next.gep1
; CHECK: load <2 x i32>, <2 x i32>* [[LOAD_PTR]]
; CHECK: load i32, i32* %next.gep
; CHECK: [[STORE_PTR:%[0-9]+]] = bitcast i32* %next.gep
; CHECK: store <2 x i32>
; CHECK-SAME: <2 x i32>* [[STORE_PTR]]
; CHECK: load i32, i32* %next.gep1
define void @correct_order(i32* noalias %ptr) {
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

