; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; Test that we order instruction within a packet correctly. In this case,
; we added a definition of a value after the use in a packet, which
; caused an assert.

define void @f0(i32 %a0) {
b0:
  %v0 = ashr i32 %a0, 1
  br i1 undef, label %b3, label %b1

b1:                                               ; preds = %b1, %b0
  %v1 = phi i32 [ %v23, %b1 ], [ undef, %b0 ]
  %v2 = phi i64 [ %v14, %b1 ], [ 0, %b0 ]
  %v3 = phi i64 [ %v11, %b1 ], [ 0, %b0 ]
  %v4 = phi i32 [ %v25, %b1 ], [ 0, %b0 ]
  %v5 = phi i32 [ %v6, %b1 ], [ undef, %b0 ]
  %v6 = phi i32 [ %v20, %b1 ], [ undef, %b0 ]
  %v7 = phi i32 [ %v24, %b1 ], [ undef, %b0 ]
  %v8 = tail call i32 @llvm.hexagon.A2.combine.lh(i32 %v6, i32 %v5)
  %v9 = tail call i64 @llvm.hexagon.A2.combinew(i32 %v8, i32 undef)
  %v10 = tail call i64 @llvm.hexagon.A2.combinew(i32 %v1, i32 undef)
  %v11 = tail call i64 @llvm.hexagon.M2.vdmacs.s0(i64 %v3, i64 %v9, i64 undef)
  %v12 = tail call i64 @llvm.hexagon.A2.combinew(i32 %v5, i32 %v5)
  %v13 = tail call i64 @llvm.hexagon.S2.valignib(i64 %v10, i64 undef, i32 2)
  %v14 = tail call i64 @llvm.hexagon.M2.vdmacs.s0(i64 %v2, i64 %v12, i64 %v13)
  %v15 = inttoptr i32 %v7 to i16*
  %v16 = load i16, i16* %v15, align 2
  %v17 = sext i16 %v16 to i32
  %v18 = add nsw i32 %v7, -8
  %v19 = tail call i64 @llvm.hexagon.M2.vdmacs.s0(i64 undef, i64 %v12, i64 0)
  %v20 = tail call i32 @llvm.hexagon.A2.combine.ll(i32 %v17, i32 %v1)
  %v21 = inttoptr i32 %v18 to i16*
  %v22 = load i16, i16* %v21, align 2
  %v23 = sext i16 %v22 to i32
  %v24 = add nsw i32 %v7, -16
  %v25 = add nsw i32 %v4, 1
  %v26 = icmp eq i32 %v25, %v0
  br i1 %v26, label %b2, label %b1

b2:                                               ; preds = %b1
  %v27 = phi i64 [ %v19, %b1 ]
  %v28 = phi i64 [ %v14, %b1 ]
  %v29 = phi i64 [ %v11, %b1 ]
  %v30 = trunc i64 %v27 to i32
  %v31 = trunc i64 %v28 to i32
  %v32 = lshr i64 %v29, 32
  %v33 = trunc i64 %v32 to i32
  br label %b3

b3:                                               ; preds = %b2, %b0
  %v34 = phi i32 [ %v30, %b2 ], [ undef, %b0 ]
  %v35 = phi i32 [ %v31, %b2 ], [ undef, %b0 ]
  %v36 = phi i32 [ %v33, %b2 ], [ undef, %b0 ]
  %v37 = bitcast i8* undef to i32*
  store i32 %v35, i32* %v37, align 4
  %v38 = getelementptr inbounds i8, i8* null, i32 8
  %v39 = bitcast i8* %v38 to i32*
  store i32 %v34, i32* %v39, align 4
  %v40 = bitcast i8* undef to i32*
  store i32 %v36, i32* %v40, align 4
  call void @llvm.trap()
  unreachable
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.combine.ll(i32, i32) #0

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.A2.combinew(i32, i32) #0

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M2.vdmacs.s0(i64, i64, i64) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.combine.lh(i32, i32) #0

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.valignib(i64, i64, i32) #0

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #1

attributes #0 = { nounwind readnone }
attributes #1 = { noreturn nounwind }
