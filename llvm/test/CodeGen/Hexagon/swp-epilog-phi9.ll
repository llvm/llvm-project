; RUN: llc -march=hexagon -hexagon-initial-cfg-cleanup=0 < %s | FileCheck %s

; Test that we generate the correct Phi name in the last couple of epilog
; blocks, when there are 3 epilog blocks. The Phi was scheduled in stage
; 2, so the computation for the number of Phis needs to be adjusted when
; the incoming prolog block is from prolog 0 or prolog 1.
; Note: the pipeliner no longer generates a 3 stage pipeline for this test.

; CHECK: loop0
; CHECK: [[REG0:r([0-9]+)]] = add(r{{[0-8]+}},#8)
; CHECK: endloop0
; CHECK: [[REG0]] = add(r{{[0-9]+}},#8)

; Function Attrs: nounwind
define void @f0(i16* nocapture readonly %a0) #0 {
b0:
  %v0 = alloca [129 x i32], align 8
  br i1 undef, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i16* [ %a0, %b1 ], [ %v2, %b2 ]
  %v2 = phi i16* [ undef, %b1 ], [ %v15, %b2 ]
  %v3 = phi i32* [ null, %b1 ], [ %v4, %b2 ]
  %v4 = phi i32* [ null, %b1 ], [ %v14, %b2 ]
  %v5 = phi i32 [ 0, %b1 ], [ %v13, %b2 ]
  %v6 = phi i16* [ undef, %b1 ], [ %v12, %b2 ]
  %v7 = load i16, i16* %v2, align 2
  %v8 = sext i16 %v7 to i32
  %v9 = call i32 @llvm.hexagon.M2.mpy.ll.s0(i32 %v8, i32 %v8) #2
  %v10 = load i16, i16* %v6, align 2
  %v11 = call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s0(i32 %v9, i32 undef, i32 undef) #2
  store i32 %v11, i32* %v4, align 4
  %v12 = getelementptr inbounds i16, i16* %v6, i32 -1
  %v13 = add i32 %v5, 1
  %v14 = getelementptr inbounds i32, i32* %v3, i32 2
  %v15 = getelementptr inbounds i16, i16* %v1, i32 2
  %v16 = icmp slt i32 %v13, undef
  br i1 %v16, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  unreachable
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M2.mpy.ll.s0(i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s0(i32, i32, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
