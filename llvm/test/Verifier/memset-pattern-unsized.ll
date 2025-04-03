; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

; CHECK: unsized types cannot be used as memset patterns

%X = type opaque
define void @bar(ptr %P, %X %value) {
  call void @llvm.experimental.memset.pattern.p0.s_s.i32.0(ptr %P, %X %value, i32 4, i1 false)
  ret void
}
declare void @llvm.experimental.memset.pattern.p0.s_s.i32.0(ptr nocapture, %X, i32, i1) nounwind
