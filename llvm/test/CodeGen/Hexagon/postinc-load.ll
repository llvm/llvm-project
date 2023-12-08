; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that post-increment load instructions are being generated.
; CHECK: r{{[0-9]+}} = memw(r{{[0-9]+}}++#4)

define i32 @f0(ptr nocapture %a0, ptr nocapture %a1, i32 %a2) #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ %v11, %b1 ], [ 10, %b0 ]
  %v1 = phi ptr [ %a0, %b0 ], [ %v9, %b1 ]
  %v2 = phi ptr [ %a1, %b0 ], [ %v10, %b1 ]
  %v3 = phi i32 [ 0, %b0 ], [ %v8, %b1 ]
  %v4 = load i32, ptr %v1, align 4
  %v5 = load i16, ptr %v2, align 2
  %v6 = sext i16 %v5 to i32
  %v7 = add i32 %v4, %v3
  %v8 = add i32 %v7, %v6
  %v9 = getelementptr i32, ptr %v1, i32 1
  %v10 = getelementptr i16, ptr %v2, i32 1
  %v11 = add i32 %v0, -1
  %v12 = icmp eq i32 %v11, 0
  br i1 %v12, label %b2, label %b1

b2:                                               ; preds = %b1
  ret i32 %v8
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
