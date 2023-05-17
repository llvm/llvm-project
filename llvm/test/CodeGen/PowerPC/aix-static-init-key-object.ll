; RUN: not --crash llc -mtriple powerpc-ibm-aix-xcoff < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple powerpc64-ibm-aix-xcoff < %s 2>&1 | FileCheck %s

@v = global i8 0

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo, ptr @v}]

define void @foo() {
  ret void
}

; CHECK: LLVM ERROR: associated data of XXStructor list is not yet supported on AIX
