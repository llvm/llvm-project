; RUN: sed -e "s,CC,cfguard_checkcc,g" %s | not --crash llc -mtriple=arm64-apple-darwin -o - 2>&1 | FileCheck %s --check-prefix=CFGUARD
; RUN: sed -e "s,CC,aarch64_sve_vector_pcs,g" %s | not --crash llc -mtriple=arm64-apple-darwin -o - 2>&1 | FileCheck %s --check-prefix=SVE_VECTOR_PCS

define CC void @f0() {
  unreachable
}

; CFGUARD: Calling convention CFGuard_Check is unsupported on Darwin.
; SVE_VECTOR_PCS: Calling convention SVE_VectorCall is unsupported on Darwin.
