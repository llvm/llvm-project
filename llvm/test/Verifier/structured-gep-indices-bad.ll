; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

%S = type { i32, i32 }

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

define void @fiz(ptr %src) {
entry:
; CHECK: Index operand type must be an integer
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([ 2 x i32 ]) %src, float 1.0)
  ret void
}

define void @fuz(ptr %src) {
entry:
; CHECK: Intrinsic first parameter is missing an ElementType attribute
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr %src, i32 0)
  ret void
}
