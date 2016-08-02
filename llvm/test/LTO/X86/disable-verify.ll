; RUN: llvm-as < %s >%t.bc
; RUN: llvm-lto -debug-pass=Arguments -exported-symbol=_f -o /dev/null %t.bc 2>&1 -disable-verify | FileCheck %s
; RUN: llvm-lto -debug-pass=Arguments -exported-symbol=_f -o /dev/null %t.bc 2>&1 | FileCheck %s -check-prefix=VERIFY

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; -disable-verify should disable verification from the optimization pipeline.
; CHECK: Pass Arguments: -verify -internalize
; CHECK-NOT: -verify

; VERIFY: Pass Arguments: -verify -internalize
; VERIFY: Pass Arguments: {{.*}} -verify {{.*}} -verify

define void @f() {
entry:
  ret void
}
