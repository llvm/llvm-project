; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that we are able to predicate instructions.

; CHECK: if ({{!?}}p{{[0-3]}}{{(.new)?}}) r{{[0-9]+}} = {{and|aslh}}
; CHECK: if ({{!?}}p{{[0-3]}}{{(.new)?}}) r{{[0-9]+}} = {{and|aslh}}

@g0 = external global i32
@g1 = external global i32

define i32 @f0(i8 zeroext %a0, i8 zeroext %a1) #0 {
b0:
  %v0 = icmp eq i8 %a0, %a1
  br i1 %v0, label %b1, label %b2

b1:                                               ; preds = %b0
  %v1 = zext i8 %a0 to i32
  %v2 = shl nuw nsw i32 %v1, 16
  br label %b3

b2:                                               ; preds = %b0
  %v3 = and i8 %a1, %a0
  %v4 = zext i8 %v3 to i32
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v5 = phi i32 [ %v4, %b2 ], [ %v2, %b1 ]
  store i32 %v5, i32* @g0, align 4
  %v6 = load i32, i32* @g1, align 4
  ret i32 %v6
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
