; RUN: llc -march=hexagon -O3 -verify-machineinstrs < %s | FileCheck %s
; REQUIRES: asserts

; Check for sane output. This testcase used to crash.
; CHECK: jumpr r31

target triple = "hexagon"

@g0 = external hidden unnamed_addr constant [9 x i16], align 8

; Function Attrs: nounwind readnone
define i64 @fred(i32 %a0) local_unnamed_addr #0 {
b1:
  %v2 = icmp slt i32 %a0, 1
  br i1 %v2, label %b26, label %b3

b3:                                               ; preds = %b1
  %v4 = tail call i32 @llvm.hexagon.S2.clb(i32 %a0)
  %v5 = add nsw i32 %v4, -12
  %v6 = add nsw i32 %v4, -28
  %v7 = tail call i32 @llvm.hexagon.S2.asl.r.r(i32 %a0, i32 %v6)
  %v8 = add nsw i32 %v7, -8
  %v9 = tail call i32 @llvm.hexagon.S2.asl.r.r(i32 %a0, i32 %v5)
  %v10 = getelementptr inbounds [9 x i16], ptr @g0, i32 0, i32 %v8
  %v11 = load i16, ptr %v10, align 2
  %v12 = sext i16 %v11 to i32
  %v13 = shl nsw i32 %v12, 16
  %v14 = add nsw i32 %v7, -7
  %v15 = getelementptr inbounds [9 x i16], ptr @g0, i32 0, i32 %v14
  %v16 = load i16, ptr %v15, align 2
  %v17 = sub i16 %v11, %v16
  %v18 = and i32 %v9, 65535
  %v19 = zext i16 %v17 to i32
  %v20 = tail call i32 @llvm.hexagon.M2.mpyu.nac.ll.s0(i32 %v13, i32 %v18, i32 %v19) #1
  %v21 = add nsw i32 %v4, -32
  %v22 = zext i32 %v21 to i64
  %v23 = shl nuw i64 %v22, 32
  %v24 = zext i32 %v20 to i64
  %v25 = or i64 %v23, %v24
  br label %b26

b26:                                              ; preds = %b3, %b1
  %v27 = phi i64 [ %v25, %b3 ], [ 2147483648, %b1 ]
  ret i64 %v27
}

declare i32 @llvm.hexagon.S2.clb(i32) #1
declare i32 @llvm.hexagon.S2.asl.r.r(i32, i32) #1
declare i32 @llvm.hexagon.M2.mpyu.nac.ll.s0(i32, i32, i32) #1

attributes #0 = { nounwind readnone "target-cpu"="hexagonv55" "target-features"="-hvx,-long-calls" }
attributes #1 = { nounwind readnone }
