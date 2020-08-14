; RUN: not --crash llc -mtriple powerpc-ibm-aix-xcoff -o - %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple powerpc64-ibm-aix-xcoff -o - %s 2>&1 | FileCheck %s

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @foo, i8* null }]

define internal void @foo() {
  ret void
}

; CHECK: LLVM ERROR: cannot produce a unique identifier for this module based on strong external symbols
