; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff < \
; RUN: %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff < \
; RUN: %s | FileCheck %s

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo, ptr null }]
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @bar, ptr null }]

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
