; RUN: llvm-profdata merge %S/Inputs/cs_vs_nocs.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-warn-missing-function=true -pgo-test-profile-file=%t.profdata -S 2>&1 | FileCheck %s

; CHECK: warning: {{.+}}: no profile data available for function bar
; CHECK-NOT: warning: {{.+}}: function control flow change detected (hash mismatch)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @bar() {
entry:
  ret i32 0
}
