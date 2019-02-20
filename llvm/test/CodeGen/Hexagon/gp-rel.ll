; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that gp-relative instructions are being generated.

; CHECK: r{{[0-9]+}} = memw(gp+#g0)
; CHECK: r{{[0-9]+}} = memw(gp+#g1)
; CHECK: if (p{{[0-3]}}) memw(##g2) = r{{[0-9]+}}

@g0 = common global i32 0, align 4
@g1 = common global i32 0, align 4
@g2 = common global i32 0, align 4

define i32 @f0(i32 %a0) #0 {
b0:
  %v0 = load i32, i32* @g0, align 4
  %v1 = load i32, i32* @g1, align 4
  %v2 = add nsw i32 %v1, %v0
  %v3 = icmp eq i32 %v0, %v1
  br i1 %v3, label %b2, label %b1

b1:                                               ; preds = %b0
  %v4 = load i32, i32* @g2, align 4
  br label %b3

b2:                                               ; preds = %b0
  %v5 = add nsw i32 %v2, %v0
  store i32 %v5, i32* @g2, align 4
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v6 = phi i32 [ %v4, %b1 ], [ %v5, %b2 ]
  %v7 = icmp eq i32 %v2, %v6
  %v8 = select i1 %v7, i32 %v6, i32 %v1
  ret i32 %v8
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
