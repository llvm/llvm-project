; RUN: opt -early-cse -earlycse-debug-hash -S -o - %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64-ni:2"
target triple = "armv7-unknown-linux-gnueabi"

%struct.anon = type { i32 }

@onesstruct = private constant %struct.anon { i32 -1 }, align 4

define i32 @allones_struct() {
; CHECK-LABEL: @allones_struct()
; CHECK-NEXT:    %1 = load [1 x i32], ptr @onesstruct, align 4
; CHECK-NEXT:    %2 = extractvalue [1 x i32] %1, 0
; CHECK-NEXT:    ret i32 %2
  %1 = load [1 x i32], ptr @onesstruct, align 4
  %2 = extractvalue [1 x i32] %1, 0
  ret i32 %2
}

define i32 @allones_int() {
; CHECK-LABEL: @allones_int()
; CHECK-NEXT:    ret i32 -1
  %1 = load i32, ptr @onesstruct, align 4
  ret i32 %1
}

define ptr @allones_ptr() {
; CHECK-LABEL: @allones_ptr()
; CHECK-NEXT:    ret ptr inttoptr (i32 -1 to ptr)
  %1 = load ptr, ptr @onesstruct, align 4
  ret ptr %1
}

define ptr addrspace(1) @allones_ptr1() {
; CHECK-LABEL: @allones_ptr1()
; CHECK-NEXT:    ret ptr addrspace(1) inttoptr (i32 -1 to ptr addrspace(1))
  %1 = load ptr addrspace(1), ptr @onesstruct, align 4
  ret ptr addrspace(1) %1
}

define ptr addrspace(2) @allones_ptr2() {
; CHECK-LABEL: @allones_ptr2()
; CHECK-NEXT:    %1 = load ptr addrspace(2), ptr @onesstruct, align 4
; CHECK-NEXT:    ret ptr addrspace(2) %1
  %1 = load ptr addrspace(2), ptr @onesstruct, align 4
  ret ptr addrspace(2) %1
}
