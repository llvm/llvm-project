target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; RUN: opt < %s -S -passes='loop-vectorize<interleave-forced-only;vectorize-forced-only>' 2>&1  | FileCheck %s
; RUN: opt < %s -S -passes='loop-vectorize<no-interleave-forced-only;no-vectorize-forced-only>' 2>&1 | FileCheck %s

; Dummy test to check LoopVectorize options.
; CHECK-LABEL: dummy
define void @dummy() {
  ret void
}
