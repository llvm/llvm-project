; RUN: llc -march=hexagon -verify-machineinstrs < %s | FileCheck %s
; REQUIRES: asserts

; Check for some output (as opposed to a crash).
; CHECK: loop0

target triple = "hexagon"

@x = external local_unnamed_addr global [80 x i32], align 8

; Function Attrs: nounwind
define void @fred() local_unnamed_addr #0 {
b0:
  br label %b1

b1:                                               ; preds = %b20, %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v3 = phi i32 [ 0, %b1 ], [ %v17, %b2 ]
  %v4 = phi i32 [ 0, %b1 ], [ %v16, %b2 ]
  %v5 = phi i32 [ undef, %b1 ], [ %v18, %b2 ]
  %v6 = load i32, i32* undef, align 8
  %v7 = icmp sgt i32 %v6, undef
  %v8 = select i1 %v7, i32 %v3, i32 %v4
  %v9 = select i1 undef, i32 0, i32 %v8
  %v10 = select i1 undef, i32 undef, i32 %v9
  %v11 = select i1 undef, i32 0, i32 %v10
  %v12 = icmp sgt i32 undef, 0
  %v13 = select i1 %v12, i32 undef, i32 %v11
  %v14 = select i1 false, i32 undef, i32 %v13
  %v15 = select i1 false, i32 undef, i32 %v14
  %v16 = select i1 false, i32 undef, i32 %v15
  %v17 = add nsw i32 %v3, 8
  %v18 = add i32 %v5, -8
  %v19 = icmp eq i32 %v18, 0
  br i1 %v19, label %b20, label %b2

b20:                                              ; preds = %b2
  %v21 = getelementptr inbounds [80 x i32], [80 x i32]* @x, i32 0, i32 %v16
  store i32 -2000, i32* %v21, align 4
  br label %b1
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" "target-features"="-hvx,-hvx-double,-long-calls" }
