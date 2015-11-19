; RUN: not opt -verify < %s 2>&1 | FileCheck %s

; CHECK: alignment argument of memory intrinsics must be a power of 2 

define void @foo(i8* %P, i8* %Q) {
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %P, i8* %Q, i32 4, i32 3, i1 false)
  ret void
}
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
