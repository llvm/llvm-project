; RUN: llc -mtriple=hexagon -hexagon-hwloop-preheader < %s | FileCheck %s

; check for lack of assertion failures.

; CHECK: %bb.0

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.sath(i32) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.neg(i32) #0

define void @f0(i16 signext %a0) {
b0:
  %v0 = icmp slt i16 %a0, 1
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b2, %b0
  %v1 = phi i16 [ %v11, %b2 ], [ %a0, %b0 ]
  %v2 = sext i16 %v1 to i32
  %v3 = tail call i32 @llvm.hexagon.A2.neg(i32 %v2)
  %v4 = tail call i32 @llvm.hexagon.A2.sath(i32 %v3)
  %v5 = trunc i32 %v4 to i16
  %v6 = shl i32 %v4, 16
  %v7 = ashr exact i32 %v6, 16
  %v8 = icmp slt i16 %v5, 0
  br i1 %v8, label %b2, label %b3

b2:                                               ; preds = %b1
  %v9 = tail call i32 @llvm.hexagon.A2.neg(i32 %v7)
  %v10 = tail call i32 @llvm.hexagon.A2.sath(i32 %v9)
  %v11 = trunc i32 %v10 to i16
  %v12 = icmp slt i16 %v11, 1
  br i1 %v12, label %b1, label %b3

b3:                                               ; preds = %b2, %b1, %b0
  ret void
}

attributes #0 = { nounwind readnone }
