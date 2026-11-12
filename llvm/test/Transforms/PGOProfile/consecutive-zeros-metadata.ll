; RUN: llvm-profdata merge %S/Inputs/consecutive-zeros-metadata.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s

;; Check that if we have multiple external symbols for indirect call value
;; profile information, we concatenate the zeros.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @func1() {
entry:
  ret void
}

define void @test_call(ptr %fptr) {
entry:
  call void %fptr()
  ret void
}

; CHECK: !{!"VP", i32 0, i64 130, i64 -2545542355363006406, i64 80, i64 0, i64 50}
