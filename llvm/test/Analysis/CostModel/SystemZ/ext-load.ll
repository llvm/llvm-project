; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s
;
; Test that an extension of a load does not get an additional cost in cases
; where the load performs the extension.

define void @sext() {
  %li8 = load i8, ptr undef
  sext i8 %li8 to i16
  sext i8 %li8 to i32
  sext i8 %li8 to i64

  %li16 = load i16, ptr undef
  sext i16 %li16 to i32
  sext i16 %li16 to i64

  %li32 = load i32, ptr undef
  sext i32 %li32 to i64

  ret void

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li8 = load i8, ptr undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %1 = sext i8 %li8 to i16
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %2 = sext i8 %li8 to i32
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %3 = sext i8 %li8 to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li16 = load i16, ptr undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %4 = sext i16 %li16 to i32
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %5 = sext i16 %li16 to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32 = load i32, ptr undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %6 = sext i32 %li32 to i64
}

define void @zext() {
  %li8 = load i8, ptr undef
  zext i8 %li8 to i16
  zext i8 %li8 to i32
  zext i8 %li8 to i64

  %li16 = load i16, ptr undef
  zext i16 %li16 to i32
  zext i16 %li16 to i64

  %li32 = load i32, ptr undef
  zext i32 %li32 to i64

  ret void

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li8 = load i8, ptr undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %1 = zext i8 %li8 to i16
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %2 = zext i8 %li8 to i32
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %3 = zext i8 %li8 to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li16 = load i16, ptr undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %4 = zext i16 %li16 to i32
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %5 = zext i16 %li16 to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32 = load i32, ptr undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %6 = zext i32 %li32 to i64
}
