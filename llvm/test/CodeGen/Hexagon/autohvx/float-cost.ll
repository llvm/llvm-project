; RUN: opt -march=hexagon -passes=loop-vectorize -hexagon-autohvx -debug-only=loop-vectorize -S < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; Check that the cost model makes vectorization non-profitable.
; CHECK: LV: Vectorization is possible but not beneficial

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define void @f0(i8* nocapture readonly %a0, i8* nocapture %a1, i32 %a2, i32 %a3, i32 %a4, float %a5, float %a6) #0 {
b0:
  %v0 = icmp sgt i32 %a2, 0
  br i1 %v0, label %b1, label %b2

b1:                                               ; preds = %b0
  %v1 = add nsw i32 %a3, -1
  %v2 = sitofp i32 %v1 to float
  %v3 = fcmp olt float %v2, %a6
  %v4 = select i1 %v3, float %v2, float %a6
  %v5 = sitofp i32 %a4 to float
  %v6 = fmul float %v4, %v5
  %v7 = sitofp i32 %a2 to float
  %v8 = fmul float %v6, %v7
  %v9 = add nsw i32 %a4, -1
  %v10 = sitofp i32 %v9 to float
  %v11 = fcmp olt float %v10, %a5
  %v12 = select i1 %v11, float %v10, float %a5
  %v13 = fmul float %v12, %v7
  %v14 = fadd float %v13, %v8
  %v15 = fptosi float %v14 to i32
  %v16 = fadd float %a5, 1.000000e+00
  %v17 = fcmp ogt float %v16, %v10
  %v18 = select i1 %v17, float %v10, float %v16
  %v19 = fmul float %v18, %v7
  %v20 = fadd float %v19, %v8
  %v21 = fptosi float %v20 to i32
  %v22 = fadd float %a6, 1.000000e+00
  %v23 = fcmp ogt float %v22, %v2
  %v24 = select i1 %v23, float %v2, float %v22
  %v25 = fmul float %v24, %v5
  %v26 = fmul float %v25, %v7
  %v27 = fadd float %v13, %v26
  %v28 = fptosi float %v27 to i32
  %v29 = fadd float %v19, %v26
  %v30 = fptosi float %v29 to i32
  br label %b3

b2:                                               ; preds = %b3, %b0
  ret void

b3:                                               ; preds = %b3, %b1
  %v31 = phi i32 [ 0, %b1 ], [ %v60, %b3 ]
  %v32 = add nsw i32 %v31, %v15
  %v33 = getelementptr inbounds i8, i8* %a0, i32 %v32
  %v34 = load i8, i8* %v33, align 1, !tbaa !0
  %v35 = add nsw i32 %v31, %v21
  %v36 = getelementptr inbounds i8, i8* %a0, i32 %v35
  %v37 = load i8, i8* %v36, align 1, !tbaa !0
  %v38 = add nsw i32 %v31, %v28
  %v39 = getelementptr inbounds i8, i8* %a0, i32 %v38
  %v40 = load i8, i8* %v39, align 1, !tbaa !0
  %v41 = add nsw i32 %v31, %v30
  %v42 = getelementptr inbounds i8, i8* %a0, i32 %v41
  %v43 = load i8, i8* %v42, align 1, !tbaa !0
  %v44 = uitofp i8 %v34 to float
  %v45 = uitofp i8 %v37 to float
  %v46 = uitofp i8 %v40 to float
  %v47 = uitofp i8 %v43 to float
  %v48 = fsub float %v45, %v44
  %v49 = fmul float %v48, 0x3FD99999A0000000
  %v50 = fadd float %v49, %v44
  %v51 = fsub float %v47, %v46
  %v52 = fmul float %v51, 0x3FD99999A0000000
  %v53 = fadd float %v52, %v46
  %v54 = fsub float %v53, %v50
  %v55 = fmul float %v54, 0x3FD99999A0000000
  %v56 = fadd float %v50, %v55
  %v57 = fadd float %v56, 5.000000e-01
  %v58 = fptoui float %v57 to i8
  %v59 = getelementptr inbounds i8, i8* %a1, i32 %v31
  store i8 %v58, i8* %v59, align 1, !tbaa !0
  %v60 = add nuw nsw i32 %v31, 1
  %v61 = icmp eq i32 %v60, %a2
  br i1 %v61, label %b2, label %b3
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv65" "target-features"="+hvx-length128b,+hvxv65" }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
