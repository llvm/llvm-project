;; Ensure that SHF_ALLOC section flag is not set for the __llvm_covmap section on Linux.
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

@covfun = linkonce_odr hidden constant i32 0, section "__llvm_covfun"
@__llvm_coverage_mapping = internal constant i32 0, section "__llvm_covmap"

; CHECK-DAG: .section	__llvm_covfun,""
; CHECK-DAG: .section	__llvm_covmap,""
