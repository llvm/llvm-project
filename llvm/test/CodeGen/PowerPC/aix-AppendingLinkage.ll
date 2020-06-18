; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc-ibm-aix-xcoff < \
; RUN: %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc64-ibm-aix-xcoff < \
; RUN: %s | FileCheck %s

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @foo, i8* null }]
@llvm.global_dtors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @bar, i8* null }]

define dso_local void @foo() {
entry:
  ret void
}

define dso_local void @bar() {
entry:
  ret void
}

; CHECK-NOT: llvm.global_ctors
; CHECK-NOT: llvm.global_dtors
