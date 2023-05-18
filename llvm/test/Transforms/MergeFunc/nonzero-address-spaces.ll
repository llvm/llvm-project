; RUN: opt -S -passes=mergefunc < %s | FileCheck %s

; MergeFunctions should respect the default function address
; space specified in the data layout.

target datalayout = "e-P1-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare void @stuff()

; CHECK-LABEL: @f0(
define void @f0(i64 %p0) {
entry:
  call void @stuff()
  call void @stuff()
  call void @stuff()
  ret void
}

; CHECK-LABEL: @f1(
; CHECK: ptrtoint ptr
; CHECK: tail call addrspace(1) void @f0(i64

define void @f1(ptr %p0) {
entry:
  call void @stuff()
  call void @stuff()
  call void @stuff()
  ret void
}

