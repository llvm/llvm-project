; RUN: llc -march=hexagon -verify-machineinstrs < %s | FileCheck %s

; Check for sane output.
; CHECK: vmpyweh

target triple = "hexagon"

declare i32 @llvm.hexagon.S2.clb(i32) #0
declare i32 @llvm.hexagon.S2.asl.r.r(i32, i32) #0
declare i32 @llvm.hexagon.S2.vrndpackwh(i64) #0
declare i64 @llvm.hexagon.M2.mmpyl.s1(i64, i64) #0

define i64 @fred(i32 %a0, i32 %a1) local_unnamed_addr #1 {
b2:
  br i1 undef, label %b15, label %b3

b3:                                               ; preds = %b2
  %v4 = tail call i32 @llvm.hexagon.S2.clb(i32 %a1) #0
  %v5 = add nsw i32 %v4, -32
  %v6 = zext i32 %v5 to i64
  %v7 = shl nuw i64 %v6, 32
  %v8 = or i64 %v7, 0
  %v9 = tail call i32 @llvm.hexagon.S2.asl.r.r(i32 %a0, i32 0)
  %v10 = tail call i32 @llvm.hexagon.S2.vrndpackwh(i64 %v8)
  %v11 = sext i32 %v9 to i64
  %v12 = sext i32 %v10 to i64
  %v13 = tail call i64 @llvm.hexagon.M2.mmpyl.s1(i64 %v11, i64 %v12)
  %v14 = and i64 %v13, 4294967295
  br label %b15

b15:                                              ; preds = %b3, %b2
  %v16 = phi i64 [ %v14, %b3 ], [ 0, %b2 ]
  %v17 = or i64 0, %v16
  ret i64 %v17
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "target-cpu"="hexagonv55" }
