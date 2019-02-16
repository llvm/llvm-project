; RUN: llc -march=hexagon -O0 < %s

; This is a regression test which makes sure that the offset check
; is available for STRiw_indexed instruction. This is required
; by 'Hexagon Expand Predicate Spill Code' pass.

define i32 @f0(i32 %a0, i32 %a1) #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i32, align 4
  %v2 = alloca i32, align 4
  store i32 %a0, i32* %v1, align 4
  store i32 %a1, i32* %v2, align 4
  %v3 = load i32, i32* %v1, align 4
  %v4 = load i32, i32* %v2, align 4
  %v5 = icmp sgt i32 %v3, %v4
  br i1 %v5, label %b1, label %b2

b1:                                               ; preds = %b0
  %v6 = load i32, i32* %v1, align 4
  %v7 = load i32, i32* %v2, align 4
  %v8 = add nsw i32 %v6, %v7
  store i32 %v8, i32* %v0
  br label %b3

b2:                                               ; preds = %b0
  %v9 = load i32, i32* %v1, align 4
  %v10 = load i32, i32* %v2, align 4
  %v11 = sub nsw i32 %v9, %v10
  store i32 %v11, i32* %v0
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v12 = load i32, i32* %v0
  ret i32 %v12
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
