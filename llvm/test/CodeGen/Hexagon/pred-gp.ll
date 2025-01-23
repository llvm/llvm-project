; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; Check that we are able to predicate instructions with gp-relative
; addressing mode.

; CHECK: if ({{!?}}p{{[0-3]+}}{{(.new)?}}) r{{[0-9]+}} = memw(##g{{[01]}})
; CHECK: if ({{!?}}p{{[0-3]+}}) r{{[0-9]+}} = memw(##g{{[01]}})

@g0 = external global i32
@g1 = common global i32 0, align 4

define i32 @f0(i8 zeroext %a0, i8 zeroext %a1) #0 {
b0:
  %v0 = icmp eq i8 %a0, %a1
  br i1 %v0, label %b2, label %b1

b1:                                               ; preds = %b0
  %v1 = load i32, ptr @g1, align 4
  br label %b3

b2:                                               ; preds = %b0
  %v2 = load i32, ptr @g0, align 4
  store i32 %v2, ptr @g1, align 4
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v3 = phi i32 [ %v1, %b1 ], [ %v2, %b2 ]
  ret i32 %v3
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
