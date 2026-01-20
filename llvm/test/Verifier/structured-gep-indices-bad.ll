; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-p:32:32:32-p1:16:16:16-p2:8:8:8-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n8:16:32"

%S = type { i32, i32 }

@global = global %S zeroinitializer


define void @foo(ptr %src) {
entry:
; CHECK: Reached a non-composite type with more indices to process
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%S) %src, i32 0, i32 0)
  ret void
}

define void @bar(ptr %src) {
entry:
; CHECK: Indexing in a struct should be inbounds
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%S) %src, i32 2)
  ret void
}

define void @baz(ptr %src, i32 %index) {
entry:
; CHECK: Indexing into a struct requires a constant int
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%S) %src, i32 %index)
  ret void
}

define void @biz(ptr %src, i32 %index) {
entry:
; CHECK: Indexing in an array should be inbounds
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([ 2 x i32 ]) %src, i32 2)
  ret void
}

define void @fiz(ptr %src) {
entry:
; CHECK: Index operand type must be an integer
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([ 2 x i32 ]) %src, float 1.0)
  ret void
}
