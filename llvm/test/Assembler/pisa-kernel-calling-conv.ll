; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

declare pisa_kernel void @decl()
; CHECK: declare pisa_kernel void @decl()

define pisa_kernel void @kernel() {
; CHECK-LABEL: define pisa_kernel void @kernel() {
  ret void
}
