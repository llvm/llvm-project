; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-LABEL: Begin function original_testcase
define fastcc void @original_testcase() {
top:
  %0 = alloca [1 x ptr], align 4
  ; CHECK: OpCompositeInsert
  %1 = insertvalue [1 x ptr] zeroinitializer, ptr poison, 0
  store [1 x ptr] %1, ptr %0
  ret void
}

; CHECK-LABEL: Begin function additional_testcases
define fastcc void @additional_testcases() {
top:
  %0 = alloca [2 x ptr], align 4


  ; Test with different pointer types
  ; CHECK: OpCompositeInsert
  %1 = insertvalue [1 x ptr] zeroinitializer, ptr undef, 0
  ; CHECK: OpStore
  store [1 x ptr] %1, ptr %0

  ; CHECK-NEXT: OpCompositeInsert
  %2 = insertvalue {ptr, i32} zeroinitializer, ptr poison, 0
  ; CHECK: OpStore
  store {ptr, i32} %2, ptr %0

  ; CHECK-NEXT: OpCompositeInsert
  %3 = insertvalue {ptr, ptr} undef, ptr null, 0
  ; CHECK: OpStore
  store {ptr, ptr} %3, ptr %0

  ; Test with undef aggregate
  ; CHECK-NEXT: OpCompositeInsert
  %4 = insertvalue [1 x ptr] undef, ptr undef, 0
  ; CHECK: OpStore
  store [1 x ptr] %4, ptr %0

  ret void
}
