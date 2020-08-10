; RUN: not --crash llc -mtriple powerpc-ibm-aix-xcoff < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple powerpc64-ibm-aix-xcoff < %s 2>&1 | FileCheck %s

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 655, void ()* @foo, i8* null }]

define void @foo() {
  ret void
}

; CHECK: LLVM ERROR: prioritized sinit and sterm functions are not yet supported
