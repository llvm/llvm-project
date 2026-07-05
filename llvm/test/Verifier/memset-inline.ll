; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

; CHECK: alignment is not a power of two

define void @foo(ptr %P, i8 %value) {
  call void @llvm.memset.inline.p0.i32(ptr align 3 %P, i8 %value, i32 4, i1 false)
  ret void
}
declare void @llvm.memset.inline.p0.i32(ptr nocapture, i8, i32, i1) nounwind
