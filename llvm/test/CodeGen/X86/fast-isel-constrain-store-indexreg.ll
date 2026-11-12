; RUN: llc %s -o - -verify-machineinstrs | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-unknown"

@TheArray = external global [100000 x double], align 16

; This test ensures, via the machine verifier, that the register class for the
; index of the double store is correctly constrained to not include SP.

; CHECK: movsd

define i32 @main(ptr %i, double %tmpv) {
bb:
  br label %bb7

bb7:                                              ; preds = %bb7, %bb
  %storemerge = phi i32 [ 0, %bb ], [ %tmp19, %bb7 ]
  %tmp15 = zext i32 %storemerge to i64
  %tmp16 = getelementptr inbounds [100000 x double], ptr @TheArray, i64 0, i64 %tmp15
  store double %tmpv, ptr %tmp16, align 8
  %tmp18 = load i32, ptr %i, align 4
  %tmp19 = add i32 %tmp18, 1
  br label %bb7
}
