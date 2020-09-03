; Check that the GHC calling convention works (s390x)
; Thread local storage is not supported in GHC calling convention
;
; RUN: not --crash llc -mtriple=s390x-ibm-linux < %s 2>&1 | FileCheck %s

@x = thread_local global i32 0

define ghccc void @foo() nounwind {
entry:
  call void @bar(i32 *@x)
  ret void
}

declare void @bar(i32*)

; CHECK: LLVM ERROR: In GHC calling convention TLS is not supported
