; Check that the GHC calling convention works (s390x)
; In GHC calling convention a frame pointer is not supported
;
; RUN: not llc -mtriple=s390x-ibm-linux < %s 2>&1 | FileCheck %s

define ghccc void @foo(i64 %0) nounwind {
entry:
  alloca i64, i64 %0
  ret void
}

; CHECK: LLVM ERROR: In GHC calling convention a frame pointer is not supported
