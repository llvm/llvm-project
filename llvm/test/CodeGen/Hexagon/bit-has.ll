; RUN: llc -march=hexagon < %s | FileCheck %s
; REQUIRES: asserts

; This used to crash. Check for some sane output.
; CHECK: sath

target triple = "hexagon"

define void @fred() local_unnamed_addr #0 {
b0:
  %v1 = load i32, i32* undef, align 4
  %v2 = tail call i32 @llvm.hexagon.A2.sath(i32 undef)
  %v3 = and i32 %v1, 603979776
  %v4 = trunc i32 %v3 to i30
  switch i30 %v4, label %b22 [
    i30 -536870912, label %b5
    i30 -469762048, label %b6
  ]

b5:                                               ; preds = %b0
  unreachable

b6:                                               ; preds = %b0
  %v7 = load i32, i32* undef, align 4
  %v8 = sub nsw i32 65536, %v7
  %v9 = load i32, i32* undef, align 4
  %v10 = mul nsw i32 %v9, %v9
  %v11 = zext i32 %v10 to i64
  %v12 = mul nsw i32 %v2, %v8
  %v13 = sext i32 %v12 to i64
  %v14 = mul nsw i64 %v13, %v11
  %v15 = trunc i64 %v14 to i32
  %v16 = and i32 %v15, 2147483647
  store i32 %v16, i32* undef, align 4
  %v17 = lshr i64 %v14, 31
  %v18 = trunc i64 %v17 to i32
  store i32 %v18, i32* undef, align 4
  br label %b19

b19:                                              ; preds = %b6
  br i1 undef, label %b20, label %b21

b20:                                              ; preds = %b19
  unreachable

b21:                                              ; preds = %b19
  br label %b23

b22:                                              ; preds = %b0
  unreachable

b23:                                              ; preds = %b21
  %v24 = load i32, i32* undef, align 4
  %v25 = shl i32 %v24, 1
  %v26 = and i32 %v25, 65534
  %v27 = or i32 %v26, 0
  store i32 %v27, i32* undef, align 4
  ret void
}

declare i32 @llvm.hexagon.A2.sath(i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv5" "target-features"="-hvx,-long-calls" }
attributes #1 = { nounwind readnone }
