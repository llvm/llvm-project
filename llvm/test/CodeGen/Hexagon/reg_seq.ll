; RUN: llc -march=hexagon -O2 < %s | FileCheck %s

; Generate REG_SEQUENCE instead of combine
; CHECK-NOT: combine(#0

; Function Attrs: nounwind
define void @f0(i16* nocapture readonly %a0, i16* nocapture readonly %a1, i16* nocapture %a2, i16* nocapture readonly %a3, i32 %a4) #0 {
b0:
  %v0 = lshr i32 %a4, 1
  %v1 = icmp eq i32 %v0, 0
  br i1 %v1, label %b3, label %b1

b1:                                               ; preds = %b0
  %v2 = bitcast i16* %a2 to i64*
  %v3 = bitcast i16* %a1 to i64*
  %v4 = bitcast i16* %a0 to i64*
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v5 = phi i32 [ 0, %b1 ], [ %v71, %b2 ]
  %v6 = phi i64* [ %v4, %b1 ], [ %v9, %b2 ]
  %v7 = phi i64* [ %v3, %b1 ], [ %v11, %b2 ]
  %v8 = phi i64* [ %v2, %b1 ], [ %v70, %b2 ]
  %v9 = getelementptr inbounds i64, i64* %v6, i32 1
  %v10 = load i64, i64* %v6, align 8, !tbaa !0
  %v11 = getelementptr inbounds i64, i64* %v7, i32 1
  %v12 = load i64, i64* %v7, align 8, !tbaa !0
  %v13 = trunc i64 %v10 to i32
  %v14 = lshr i64 %v10, 32
  %v15 = tail call i64 @llvm.hexagon.S2.vzxthw(i32 %v13)
  %v16 = trunc i64 %v12 to i32
  %v17 = lshr i64 %v12, 32
  %v18 = tail call i64 @llvm.hexagon.S2.vzxthw(i32 %v16)
  %v19 = trunc i64 %v15 to i32
  %v20 = lshr i64 %v15, 32
  %v21 = getelementptr inbounds i16, i16* %a3, i32 %v19
  %v22 = load i16, i16* %v21, align 2, !tbaa !3
  %v23 = trunc i64 %v20 to i32
  %v24 = getelementptr inbounds i16, i16* %a3, i32 %v23
  %v25 = load i16, i16* %v24, align 2, !tbaa !3
  %v26 = trunc i64 %v18 to i32
  %v27 = lshr i64 %v18, 32
  %v28 = getelementptr inbounds i16, i16* %a3, i32 %v26
  %v29 = load i16, i16* %v28, align 2, !tbaa !3
  %v30 = trunc i64 %v27 to i32
  %v31 = getelementptr inbounds i16, i16* %a3, i32 %v30
  %v32 = load i16, i16* %v31, align 2, !tbaa !3
  %v33 = zext i16 %v32 to i64
  %v34 = shl nuw nsw i64 %v33, 32
  %v35 = zext i16 %v29 to i64
  %v36 = or i64 %v35, %v34
  %v37 = zext i16 %v25 to i64
  %v38 = shl nuw nsw i64 %v37, 32
  %v39 = zext i16 %v22 to i64
  %v40 = or i64 %v39, %v38
  %v41 = tail call i64 @llvm.hexagon.S2.vtrunewh(i64 %v36, i64 %v40)
  %v42 = getelementptr inbounds i64, i64* %v8, i32 1
  store i64 %v41, i64* %v8, align 8, !tbaa !0
  %v43 = trunc i64 %v14 to i32
  %v44 = tail call i64 @llvm.hexagon.S2.vzxthw(i32 %v43)
  %v45 = trunc i64 %v17 to i32
  %v46 = tail call i64 @llvm.hexagon.S2.vzxthw(i32 %v45)
  %v47 = trunc i64 %v44 to i32
  %v48 = lshr i64 %v44, 32
  %v49 = getelementptr inbounds i16, i16* %a3, i32 %v47
  %v50 = load i16, i16* %v49, align 2, !tbaa !3
  %v51 = trunc i64 %v48 to i32
  %v52 = getelementptr inbounds i16, i16* %a3, i32 %v51
  %v53 = load i16, i16* %v52, align 2, !tbaa !3
  %v54 = trunc i64 %v46 to i32
  %v55 = lshr i64 %v46, 32
  %v56 = getelementptr inbounds i16, i16* %a3, i32 %v54
  %v57 = load i16, i16* %v56, align 2, !tbaa !3
  %v58 = trunc i64 %v55 to i32
  %v59 = getelementptr inbounds i16, i16* %a3, i32 %v58
  %v60 = load i16, i16* %v59, align 2, !tbaa !3
  %v61 = zext i16 %v60 to i64
  %v62 = shl nuw nsw i64 %v61, 32
  %v63 = zext i16 %v57 to i64
  %v64 = or i64 %v63, %v62
  %v65 = zext i16 %v53 to i64
  %v66 = shl nuw nsw i64 %v65, 32
  %v67 = zext i16 %v50 to i64
  %v68 = or i64 %v67, %v66
  %v69 = tail call i64 @llvm.hexagon.S2.vtrunewh(i64 %v64, i64 %v68)
  %v70 = getelementptr inbounds i64, i64* %v8, i32 2
  store i64 %v69, i64* %v42, align 8, !tbaa !0
  %v71 = add nsw i32 %v5, 1
  %v72 = icmp ult i32 %v71, %v0
  br i1 %v72, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.vzxthw(i32) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.vtrunewh(i64, i64) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!4, !4, i64 0}
!4 = !{!"short", !1, i64 0}
