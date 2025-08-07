; RUN: sed -e "s,CC,cfguard_checkcc,g" %s | not --crash llc -mtriple=arm64-apple-darwin -o - 2>&1 | FileCheck %s --check-prefix=CFGUARD

define CC void @f0() {
  unreachable
}

; CFGUARD: Calling convention CFGuard_Check is unsupported on Darwin.
