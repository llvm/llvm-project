; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

; CHECK: pattern type must be an integral number of bytes

define void @foo(ptr %P, i31 %value) {
  call void @llvm.experimental.memset.pattern.p0.i31.i32(ptr align 4 %P, i31 %value, i32 4, i1 false)
  ret void
}
declare void @llvm.experimental.memset.pattern.p0.i31.i32(ptr nocapture, i31, i32, i1) nounwind
