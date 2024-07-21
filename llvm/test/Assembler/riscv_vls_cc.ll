; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: define riscv_vls_cc void @no_args() {
define riscv_vls_cc void @no_args() {
  ret void
}

; CHECK: define riscv_vls_cc void @byval_arg(ptr byval(i32) %0) {
define riscv_vls_cc void @byval_arg(ptr byval(i32)) {
  ret void
}
