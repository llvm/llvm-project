; RUN: llvm-as -disable-output %s

target datalayout = "e-p:32:32:32-p1:16:16:16-p2:8:8:8-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n8:16:32"

%S = type { i32, i32 }

define void @foo(ptr %src, i32 %index) {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([0 x %S]) %src, i32 %index, i32 1)
  ret void
}

define void @bar(ptr %src, i32 %index) {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([2 x %S]) %src, i32 %index, i32 1)
  ret void
}

define void @baz(ptr %src) {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([2 x %S]) %src, i32 1, i32 1)
  ret void
}

define void @biz(ptr %src) {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%S) %src, i32 0)
  ret void
}

define void @buz(ptr %src) {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([ 3 x [ 2 x i32 ] ]) %src, i32 2, i32 1)
  ret void
}

define void @boz(ptr %src) {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([ 0 x i32 ]) %src, i32 1)
  ret void
}

define void @foz(ptr %src) {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(i32) %src)
  ret void
}

define void @fiz(ptr %src) {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([ 0 x i32 ]) %src, i64 1)
  ret void
}

define void @fuz(ptr %src) {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([ 0 x i32 ]) %src, i8 1)
  ret void
}
