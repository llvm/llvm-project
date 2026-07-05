; RUN: llvm-as %s -o - -f | llvm-dis | FileCheck %s
; RUN: llvm-as %s -o - -f | verify-uselistorder

declare aarch64_vector_pcs void @aarch64_vector_pcs()
; CHECK: declare aarch64_vector_pcs void @aarch64_vector_pcs

define void @call_aarch64_vector_pcs() {
; CHECK: call aarch64_vector_pcs void @aarch64_vector_pcs
  call aarch64_vector_pcs void @aarch64_vector_pcs()
  ret void
}
