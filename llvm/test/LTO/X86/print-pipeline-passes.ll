; RUN: llvm-as < %s >%t.bc
; RUN: llvm-lto -print-pipeline-passes -exported-symbol=_f -o /dev/null %t.bc 2>&1 | FileCheck %s

; CHECK: pipeline-passes: verify,{{.*}},verify

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"


define void @f() {
entry:
  ret void
}
