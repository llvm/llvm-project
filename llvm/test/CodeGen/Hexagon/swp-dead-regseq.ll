; RUN: llc -march=hexagon -enable-pipeliner < %s
; REQUIRES: asserts

; Check that a dead REG_SEQUENCE doesn't ICE.

; Function Attrs: nounwind
define void @f0(i32* nocapture %a0, i32 %a1) #0 {
b0:
  %v0 = mul nsw i32 %a1, 4
  %v1 = icmp sgt i32 %v0, 0
  br i1 %v1, label %b1, label %b2

b1:                                               ; preds = %b1, %b0
  %v2 = phi i32 [ %v11, %b1 ], [ 0, %b0 ]
  %v3 = load i32, i32* null, align 4
  %v4 = zext i32 %v3 to i64
  %v5 = getelementptr inbounds i32, i32* %a0, i32 0
  %v6 = load i32, i32* %v5, align 4
  %v7 = zext i32 %v6 to i64
  %v8 = shl nuw i64 %v7, 32
  %v9 = or i64 %v8, %v4
  %v10 = tail call i64 @llvm.hexagon.M2.vdmacs.s0(i64 0, i64 %v9, i64 %v9)
  %v11 = add nsw i32 %v2, 4
  %v12 = icmp slt i32 %v11, %v0
  br i1 %v12, label %b1, label %b2

b2:                                               ; preds = %b1, %b0
  %v13 = phi i64 [ 0, %b0 ], [ %v10, %b1 ]
  %v14 = tail call i64 @llvm.hexagon.S2.asr.r.vw(i64 %v13, i32 6)
  store i64 %v14, i64* null, align 8
  unreachable
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M2.vdmacs.s0(i64, i64, i64) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.asr.r.vw(i64, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
