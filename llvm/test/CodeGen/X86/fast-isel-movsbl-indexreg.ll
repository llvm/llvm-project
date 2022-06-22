; RUN: llc %s -o - -verify-machineinstrs -fast-isel=true | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; The index register on the folded memory operand was incorrect.  Ensure we generate
; movsbl in fast-isel, but also that it passes verification which will check the register
; class.

; CHECK: movsbl

@table = external hidden global [64 x i8], align 16

define i32 @test(i32 %x, i64 %offset) {
bb:
  %tmp37 = getelementptr inbounds [64 x i8], ptr @table, i64 0, i64 %offset
  %tmp38 = load i8, ptr %tmp37, align 1
  %tmp39 = sext i8 %tmp38 to i32
  ret i32 %tmp39
}
