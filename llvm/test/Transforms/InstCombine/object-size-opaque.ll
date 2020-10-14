; RUN: opt -instcombine -S %s | FileCheck %s
%opaque = type opaque

; CHECK: call i64 @llvm.objectsize.i64
define void @foo(%opaque* sret %in, i64* %sizeptr) {
  %ptr = bitcast %opaque* %in to i8*
  %size = call i64 @llvm.objectsize.i64(i8* %ptr, i1 0, i1 0, i1 0)
  store i64 %size, i64* %sizeptr
  ret void
}

declare i64 @llvm.objectsize.i64(i8*, i1, i1, i1)
