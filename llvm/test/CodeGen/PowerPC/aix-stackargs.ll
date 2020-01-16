; RUN: not llc -mtriple powerpc-ibm-aix-xcoff < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple powerpc64-ibm-aix-xcoff < %s 2>&1 | FileCheck %s

define void @bar() {
entry:
  call void @foo(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9)
  ret void
}

declare void @foo(i32, i32, i32, i32, i32, i32, i32, i32, i32)

; CHECK: LLVM ERROR: Handling of placing parameters on the stack is unimplemented!
