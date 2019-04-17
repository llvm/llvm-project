; RUN: opt -S -mergefunc < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare void @stuff()

define void @f0(i64 %p0) {
entry:
  call void @stuff()
  call void @stuff()
  call void @stuff()
  ret void
}

; CHECK-LABEL: @f0
; CHECK:  %2 = ptrtoint i64* %0 to i64
; CHECK:  tail call void @f0(i64 %2)
; CHECK:  ret void
define void @f1(i64 addrspace(0)* %p0) {
entry:
  call void @stuff()
  call void @stuff()
  call void @stuff()
  ret void
}

