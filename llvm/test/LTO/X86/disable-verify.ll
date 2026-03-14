; RUN: llvm-as < %s >%t.bc
; RUN: llvm-lto -debug-pass-manager -exported-symbol=_f -o /dev/null %t.bc 2>&1 -disable-verify | FileCheck %s
; RUN: llvm-lto -debug-pass-manager -exported-symbol=_f -o /dev/null %t.bc 2>&1 | FileCheck %s -check-prefix=VERIFY

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; -disable-verify should disable verification from the optimization pipeline.
; CHECK-NOT: Running pass: VerifierPass

; VERIFY: Running pass: VerifierPass

define void @f() {
entry:
  ret void
}
