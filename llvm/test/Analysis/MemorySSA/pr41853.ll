; RUN: opt -S -passes='require<memoryssa>,loop-simplify,early-cse<memssa>' -earlycse-debug-hash -verify-memoryssa %s | FileCheck %s
; REQUIRES: asserts
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @func(i1 %arg)
define void @func(i1 %arg) {
  br i1 %arg, label %bb5, label %bb3

bb5:                                              ; preds = %bb5, %0
  store i16 undef, ptr undef
  br i1 %arg, label %bb5, label %bb3

bb3:                                              ; preds = %bb5, %0
  ret void
}
