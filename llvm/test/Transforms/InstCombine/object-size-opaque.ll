; RUN: opt -passes=instcombine -S %s | FileCheck %s
%opaque = type opaque

; CHECK: call i64 @llvm.objectsize.i64
define void @foo(ptr sret(%opaque) %in, ptr %sizeptr) {
  %size = call i64 @llvm.objectsize.i64(ptr %in, i1 0, i1 0, i1 0)
  store i64 %size, ptr %sizeptr
  ret void
}

declare i64 @llvm.objectsize.i64(ptr, i1, i1, i1)
