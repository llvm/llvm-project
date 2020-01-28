; RUN: not opt -verify < %s 2>&1 | FileCheck %s

; CHECK: alignment is not a power of two 

define void @foo(i8* %P, i8* %Q) {
  call void @llvm.memcpy.inline.p0i8.p0i8.i32(i8* align 3 %P, i8* %Q, i32 4, i1 false)
  ret void
}
declare void @llvm.memcpy.inline.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind
