; RUN: llvm-profdata merge %S/Inputs/diag.proftext -o %T/diag2.profdata
; RUN: opt < %s -pgo-instr-use -pgo-test-profile-file=%T/diag2.profdata -S 2>&1 | FileCheck %s

; CHECK: No profile data available for function bar

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @bar() {
entry:
  ret i32 0 
}
