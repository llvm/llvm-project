; RUN: llc -march=hexagon -O3 -verify-machineinstrs < %s | FileCheck %s
;
; Check that this testcase compiles successfully and that a new-value jump
; has been created.
; CHECK: if (cmp.gtu(r{{[0-9]+}}.new,r{{[0-9]+}})) jump

target triple = "hexagon"

@g0 = external hidden unnamed_addr global [182 x i16], align 8

define void @fred(i16 signext %a0) local_unnamed_addr #0 {
b1:
  %v2 = getelementptr inbounds [182 x i16], [182 x i16]* @g0, i32 0, i32 0
  %v3 = sext i16 %a0 to i32
  %v4 = call i32 @llvm.hexagon.A2.asrh(i32 undef)
  %v5 = trunc i32 %v4 to i16
  br i1 undef, label %b6, label %b14

b6:                                               ; preds = %b1
  %v7 = sext i16 %v5 to i32
  br label %b8

b8:                                               ; preds = %b8, %b6
  %v9 = phi i32 [ 128, %b6 ], [ %v13, %b8 ]
  %v10 = sub nsw i32 %v9, %v7
  %v11 = getelementptr inbounds [182 x i16], [182 x i16]* @g0, i32 0, i32 %v10
  %v12 = load i16, i16* %v11, align 2
  %v13 = add nuw nsw i32 %v9, 1
  br label %b8

b14:                                              ; preds = %b1
  br i1 undef, label %b16, label %b15

b15:                                              ; preds = %b14
  unreachable

b16:                                              ; preds = %b14
  %v17 = getelementptr [182 x i16], [182 x i16]* @g0, i32 0, i32 %v3
  %v18 = icmp ugt i16* %v17, %v2
  %v19 = or i1 %v18, undef
  br i1 %v19, label %b20, label %b21

b20:                                              ; preds = %b16
  unreachable

b21:                                              ; preds = %b16
  ret void
}

declare i32 @llvm.hexagon.A2.asrh(i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv62" }
attributes #1 = { nounwind readnone }
