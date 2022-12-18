; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

; CHECK: alignment is not a power of two

define void @foo(i8* %P, i8 %value) {
  call void @llvm.memset.inline.p0i8.i32(i8* align 3 %P, i8 %value, i32 4, i1 false)
  ret void
}
declare void @llvm.memset.inline.p0i8.i32(i8* nocapture, i8, i32, i1) nounwind
