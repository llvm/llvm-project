; RUN: llc -mtriple=hexagon -verify-machineinstrs -hexagon-initial-cfg-cleanup=0 < %s | FileCheck %s
; REQUIRES: asserts

; Check for some output other than crashing.
; CHECK: bitsset

target triple = "hexagon"

; Function Attrs: nounwind
define void @fred() local_unnamed_addr #0 {
b0:
  %v1 = load i32, ptr undef, align 4
  %v2 = and i32 %v1, 603979776
  %v3 = trunc i32 %v2 to i30
  switch i30 %v3, label %b23 [
    i30 -536870912, label %b4
    i30 -469762048, label %b5
  ]

b4:                                               ; preds = %b0
  unreachable

b5:                                               ; preds = %b0
  %v6 = load i32, ptr undef, align 4
  br i1 undef, label %b7, label %b8

b7:                                               ; preds = %b5
  br label %b9

b8:                                               ; preds = %b5
  br label %b9

b9:                                               ; preds = %b8, %b7
  %v10 = load i32, ptr undef, align 4
  %v11 = load i32, ptr undef, align 4
  %v12 = mul nsw i32 %v11, %v10
  %v13 = ashr i32 %v12, 13
  %v14 = mul nsw i32 %v13, %v13
  %v15 = zext i32 %v14 to i64
  %v16 = mul nsw i32 %v6, %v6
  %v17 = zext i32 %v16 to i64
  %v18 = lshr i64 %v17, 5
  %v19 = select i1 undef, i64 %v18, i64 %v17
  %v20 = mul nuw nsw i64 %v19, %v15
  %v21 = trunc i64 %v20 to i32
  %v22 = and i32 %v21, 2147483647
  store i32 %v22, ptr undef, align 4
  unreachable

b23:                                              ; preds = %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" "target-features"="-hvx,-long-calls" }
