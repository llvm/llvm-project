; RUN: opt -mtriple=hexagon -passes=loop-vectorize -hexagon-autohvx -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; Check that TTI::getMinimumVF works. The calculated MaxVF was based on the
; register pressure and was less than 64.
; CHECK: LV: Overriding calculated MaxVF({{[0-9]+}}) with target's minimum: 64

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"

%s.0 = type { ptr, i32, i32, i32, i32 }

@g0 = external dso_local local_unnamed_addr global ptr, align 4

declare void @llvm.lifetime.start.p0(ptr nocapture) #0
declare void @llvm.lifetime.end.p0(ptr nocapture) #0

; Function Attrs: nounwind
define hidden fastcc void @f0(ptr nocapture %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i8 zeroext %a5) unnamed_addr #1 {
b0:
  %v0 = alloca [4 x [9 x i16]], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %v0) #2
  %v2 = add i32 %a1, -2
  %v3 = add i32 %a3, -9
  %v4 = icmp ugt i32 %v2, %v3
  %v5 = add i32 %a2, -2
  %v6 = add i32 %a4, -9
  %v7 = icmp ugt i32 %v5, %v6
  %v8 = or i1 %v4, %v7
  %v9 = load ptr, ptr @g0, align 4, !tbaa !1
  %v10 = zext i8 %a5 to i32
  %v11 = getelementptr inbounds ptr, ptr %v9, i32 %v10
  %v12 = load ptr, ptr %v11, align 4, !tbaa !1
  %v14 = load ptr, ptr %v12, align 4, !tbaa !5
  br i1 %v8, label %b1, label %b2

b1:                                               ; preds = %b1, %b0
  %v15 = phi i32 [ 0, %b0 ], [ %v119, %b1 ]
  %v16 = add i32 %v5, %v15
  %v17 = icmp slt i32 %v16, 0
  %v18 = icmp slt i32 %v16, %a4
  %v19 = select i1 %v18, i32 %v16, i32 %v3
  %v20 = select i1 %v17, i32 0, i32 %v19
  %v21 = mul i32 %v20, %a3
  %v22 = add i32 97, %v21
  %v23 = getelementptr inbounds i8, ptr %v14, i32 %v22
  %v24 = load i8, ptr %v23, align 1, !tbaa !8
  %v25 = zext i8 %v24 to i32
  %v26 = add i32 101, %v21
  %v27 = getelementptr inbounds i8, ptr %v14, i32 %v26
  %v28 = load i8, ptr %v27, align 1, !tbaa !8
  %v29 = zext i8 %v28 to i32
  %v30 = mul nsw i32 %v29, -5
  %v31 = add nsw i32 %v30, %v25
  %v32 = add i32 106, %v21
  %v33 = getelementptr inbounds i8, ptr %v14, i32 %v32
  %v34 = load i8, ptr %v33, align 1, !tbaa !8
  %v35 = zext i8 %v34 to i32
  %v36 = mul nuw nsw i32 %v35, 20
  %v37 = add nsw i32 %v36, %v31
  %v38 = add i32 111, %v21
  %v39 = getelementptr inbounds i8, ptr %v14, i32 %v38
  %v40 = load i8, ptr %v39, align 1, !tbaa !8
  %v41 = zext i8 %v40 to i32
  %v42 = mul nuw nsw i32 %v41, 20
  %v43 = add nsw i32 %v42, %v37
  %v44 = add i32 116, %v21
  %v45 = getelementptr inbounds i8, ptr %v14, i32 %v44
  %v46 = load i8, ptr %v45, align 1, !tbaa !8
  %v47 = zext i8 %v46 to i32
  %v48 = mul nsw i32 %v47, -5
  %v49 = add nsw i32 %v48, %v43
  %v50 = add i32 120, %v21
  %v51 = getelementptr inbounds i8, ptr %v14, i32 %v50
  %v52 = load i8, ptr %v51, align 1, !tbaa !8
  %v53 = zext i8 %v52 to i32
  %v54 = add nsw i32 %v49, %v53
  %v55 = trunc i32 %v54 to i16
  %v56 = getelementptr inbounds [4 x [9 x i16]], ptr %v0, i32 0, i32 0, i32 %v15
  store i16 %v55, ptr %v56, align 2, !tbaa !9
  %v57 = mul nsw i32 %v35, -5
  %v58 = add nsw i32 %v57, %v29
  %v59 = add nsw i32 %v42, %v58
  %v60 = mul nuw nsw i32 %v47, 20
  %v61 = add nsw i32 %v60, %v59
  %v62 = mul nsw i32 %v53, -5
  %v63 = add nsw i32 %v62, %v61
  %v64 = add i32 125, %v21
  %v65 = getelementptr inbounds i8, ptr %v14, i32 %v64
  %v66 = load i8, ptr %v65, align 1, !tbaa !8
  %v67 = zext i8 %v66 to i32
  %v68 = add nsw i32 %v63, %v67
  %v69 = trunc i32 %v68 to i16
  %v70 = getelementptr inbounds [4 x [9 x i16]], ptr %v0, i32 0, i32 1, i32 %v15
  store i16 %v69, ptr %v70, align 2, !tbaa !9
  %v71 = mul nsw i32 %v41, -5
  %v72 = add nsw i32 %v71, %v35
  %v73 = add nsw i32 %v60, %v72
  %v74 = mul nuw nsw i32 %v53, 20
  %v75 = add nsw i32 %v74, %v73
  %v76 = mul nsw i32 %v67, -5
  %v77 = add nsw i32 %v76, %v75
  %v78 = add i32 130, %v21
  %v79 = getelementptr inbounds i8, ptr %v14, i32 %v78
  %v80 = load i8, ptr %v79, align 1, !tbaa !8
  %v81 = zext i8 %v80 to i32
  %v82 = add nsw i32 %v77, %v81
  %v83 = trunc i32 %v82 to i16
  %v84 = getelementptr inbounds [4 x [9 x i16]], ptr %v0, i32 0, i32 2, i32 %v15
  store i16 %v83, ptr %v84, align 2, !tbaa !9
  %v85 = add i32 92, %v21
  %v86 = getelementptr inbounds i8, ptr %v14, i32 %v85
  %v87 = load i8, ptr %v86, align 1, !tbaa !8
  %v88 = zext i8 %v87 to i16
  %v89 = add i32 135, %v21
  %v90 = getelementptr inbounds i8, ptr %v14, i32 %v89
  %v91 = load i8, ptr %v90, align 1, !tbaa !8
  %v92 = zext i8 %v91 to i16
  %v93 = mul nsw i16 %v92, -5
  %v94 = add nsw i16 %v93, %v88
  %v95 = add i32 140, %v21
  %v96 = getelementptr inbounds i8, ptr %v14, i32 %v95
  %v97 = load i8, ptr %v96, align 1, !tbaa !8
  %v98 = zext i8 %v97 to i16
  %v99 = mul nuw nsw i16 %v98, 20
  %v100 = add nsw i16 %v99, %v94
  %v101 = add i32 145, %v21
  %v102 = getelementptr inbounds i8, ptr %v14, i32 %v101
  %v103 = load i8, ptr %v102, align 1, !tbaa !8
  %v104 = zext i8 %v103 to i16
  %v105 = mul nuw nsw i16 %v104, 20
  %v106 = add i16 %v105, %v100
  %v107 = add i32 150, %v21
  %v108 = getelementptr inbounds i8, ptr %v14, i32 %v107
  %v109 = load i8, ptr %v108, align 1, !tbaa !8
  %v110 = zext i8 %v109 to i16
  %v111 = mul nsw i16 %v110, -5
  %v112 = add i16 %v111, %v106
  %v113 = add i32 154, %v21
  %v114 = getelementptr inbounds i8, ptr %v14, i32 %v113
  %v115 = load i8, ptr %v114, align 1, !tbaa !8
  %v116 = zext i8 %v115 to i16
  %v117 = add i16 %v112, %v116
  %v118 = getelementptr inbounds [4 x [9 x i16]], ptr %v0, i32 0, i32 3, i32 %v15
  store i16 %v117, ptr %v118, align 2, !tbaa !9
  %v119 = add nuw nsw i32 %v15, 1
  %v120 = icmp eq i32 %v119, 19
  br i1 %v120, label %b2, label %b1

b2:                                               ; preds = %b1, %b0
  call void @llvm.lifetime.end.p0(ptr nonnull %v0) #2
  ret void
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !2, i64 0}
!6 = !{!"", !2, i64 0, !7, i64 4, !7, i64 8, !7, i64 12, !7, i64 16}
!7 = !{!"int", !3, i64 0}
!8 = !{!3, !3, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"short", !3, i64 0}
