; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

; CHECK: alignment is not a power of two

define void @foo(ptr %P, ptr %Q) {
  call void @llvm.memcpy.p0.p0.i32(ptr align 3 %P, ptr %Q, i32 4, i1 false)
  ret void
}
declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
