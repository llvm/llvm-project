; RUN: opt -S -passes=mergefunc < %s | FileCheck %s

target datalayout = "p:32:32:32-p1:32:32:32-p2:16:16:16"

declare void @foo(i32) nounwind

; None of these functions should be merged

define i32 @store_as0(ptr %x) {
; CHECK-LABEL: @store_as0(
; CHECK: call void @foo(
  %gep = getelementptr i32, ptr %x, i32 4
  %y = load i32, ptr %gep
  call void @foo(i32 %y) nounwind
  ret i32 %y
}

define i32 @store_as1(ptr addrspace(1) %x) {
; CHECK-LABEL: @store_as1(
; CHECK: call void @foo(
  %gep = getelementptr i32, ptr addrspace(1) %x, i32 4
  %y = load i32, ptr addrspace(1) %gep
  call void @foo(i32 %y) nounwind
  ret i32 %y
}

define i32 @store_as2(ptr addrspace(2) %x) {
; CHECK-LABEL: @store_as2(
; CHECK: call void @foo(
  %gep = getelementptr i32, ptr addrspace(2) %x, i32 4
  %y = load i32, ptr addrspace(2) %gep
  call void @foo(i32 %y) nounwind
  ret i32 %y
}

