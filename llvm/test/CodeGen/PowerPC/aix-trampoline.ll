; RUN: not --crash llc -mtriple powerpc-ibm-aix-xcoff < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple powerpc64-ibm-aix-xcoff < %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: INIT_TRAMPOLINE operation is not supported on AIX.

define void @create_trampoline(ptr %buffer, ptr %nval) nounwind {
entry:
  call void @llvm.init.trampoline(ptr %buffer, ptr @nested , ptr %nval)
  ret void
}

declare i32 @nested(i32);

declare void @llvm.init.trampoline(ptr, ptr, ptr) nounwind
