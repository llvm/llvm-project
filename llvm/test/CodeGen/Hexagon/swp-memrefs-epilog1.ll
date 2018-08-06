; RUN: llc -march=hexagon -enable-pipeliner < %s | FileCheck %s

; Test that a store and load, that alias, are not put in the same packet. The
; pipeliner altered the size of the memrefs for these instructions, which
; resulted in no order dependence between the instructions in the DAG. No order
; dependence was added since the size was set to UINT_MAX, but there is a
; computation using the size that overflowed.

; CHECK: endloop0
; CHECK: memh([[REG:r([0-9]+)]]+#0) =
; CHECK: = memh([[REG]]++#2)

; Function Attrs: nounwind
define signext i16 @f0(i16* nocapture readonly %a0, i16* nocapture readonly %a1) local_unnamed_addr #0 {
b0:
  %v0 = alloca [40 x i16], align 8
  %v1 = bitcast [40 x i16]* %v0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %v1) #2
  %v2 = getelementptr inbounds [40 x i16], [40 x i16]* %v0, i32 0, i32 0
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v3 = phi i16* [ %a1, %b0 ], [ %v24, %b1 ]
  %v4 = phi i16* [ %v2, %b0 ], [ %v25, %b1 ]
  %v5 = phi i32 [ 0, %b0 ], [ %v14, %b1 ]
  %v6 = phi i32 [ 1, %b0 ], [ %v22, %b1 ]
  %v7 = phi i32 [ 0, %b0 ], [ %v23, %b1 ]
  %v8 = load i16, i16* %v3, align 2
  %v9 = sext i16 %v8 to i32
  %v10 = tail call i32 @llvm.hexagon.A2.aslh(i32 %v9)
  %v11 = tail call i32 @llvm.hexagon.S2.asr.r.r.sat(i32 %v10, i32 1)
  %v12 = tail call i32 @llvm.hexagon.A2.asrh(i32 %v11)
  %v13 = trunc i32 %v12 to i16
  store i16 %v13, i16* %v4, align 2
  %v14 = add nuw nsw i32 %v5, 1
  %v15 = icmp eq i32 %v14, 40
  %v16 = getelementptr inbounds i16, i16* %a0, i32 %v7
  %v17 = load i16, i16* %v16, align 2
  %v18 = sext i16 %v17 to i32
  %v19 = getelementptr inbounds [40 x i16], [40 x i16]* %v0, i32 0, i32 %v7
  %v20 = load i16, i16* %v19, align 2
  %v21 = sext i16 %v20 to i32
  %v22 = tail call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32 %v6, i32 %v18, i32 %v21)
  %v23 = add nuw nsw i32 %v7, 1
  %v24 = getelementptr i16, i16* %v3, i32 1
  %v25 = getelementptr i16, i16* %v4, i32 1
  br i1 %v15, label %b2, label %b1

b2:                                               ; preds = %b1
  %v26 = tail call signext i16 @f1(i32 %v22) #0
  %v27 = sext i16 %v26 to i32
  %v28 = tail call i32 @llvm.hexagon.S2.asl.r.r.sat(i32 %v22, i32 %v27)
  %v29 = tail call i32 @llvm.hexagon.A2.asrh(i32 %v28)
  %v30 = shl i32 %v29, 16
  %v31 = ashr exact i32 %v30, 16
  %v32 = icmp slt i32 %v30, 65536
  br label %b3

b3:                                               ; preds = %b2
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %v1) #2
  ret i16 0
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.asr.r.r.sat(i32, i32) #2

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.aslh(i32) #2

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.asrh(i32) #2

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32, i32, i32) #2

; Function Attrs: nounwind
declare signext i16 @f1(i32) local_unnamed_addr #0

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.asl.r.r.sat(i32, i32) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind readnone }
