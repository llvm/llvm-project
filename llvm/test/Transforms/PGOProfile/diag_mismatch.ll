; RUN: llvm-profdata merge %S/Inputs/diag.proftext -o %T/diag.profdata
; RUN: opt < %s -pgo-instr-use -pgo-test-profile-file=%T/diag.profdata -S 2>&1 | FileCheck %s

; CHECK: Function control flow change detected (hash mismatch) foo

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo() {
entry:
  ret i32 0
}
