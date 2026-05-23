; RUN: llvm-profdata merge %S/Inputs/indirect-call-vp-zeros.ll -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S

;; Check that if we have a profile with VP metadat that has only zero values
;; with zero counts, we do not emit invalid VP metadata.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test_call(ptr %fptr) {
entry:
  call void %fptr()
  ret void
}
