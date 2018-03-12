; RUN: llc -march=hexagon -O2 < %s | FileCheck %s

; Make sure we allocate less than 100 bytes of stack
; CHECK: allocframe(#{{[1-9][0-9]}}

target triple = "hexagon"

; Function Attrs: nounwind
define float @f0(float %a0) #0 {
b0:
  %v0 = alloca float, align 4
  %v1 = alloca i16, align 2
  %v2 = alloca float, align 4
  store float %a0, float* %v0, align 4, !tbaa !0
  %v3 = call signext i16 @f1(i16* %v1, float* %v0) #1
  %v4 = icmp ult i16 %v3, 3
  br i1 %v4, label %b11, label %b1

b1:                                               ; preds = %b0
  %v5 = load i16, i16* %v1, align 2, !tbaa !4
  %v6 = sext i16 %v5 to i32
  %v7 = srem i32 %v6, 3
  %v8 = icmp eq i32 %v7, 0
  br i1 %v8, label %b6, label %b2

b2:                                               ; preds = %b1
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v9 = phi i16 [ %v12, %b3 ], [ %v5, %b2 ]
  %v10 = phi i32 [ %v11, %b3 ], [ 0, %b2 ]
  %v11 = add nsw i32 %v10, -1
  %v12 = add i16 %v9, 1
  %v13 = sext i16 %v12 to i32
  %v14 = srem i32 %v13, 3
  %v15 = icmp eq i32 %v14, 0
  br i1 %v15, label %b4, label %b3

b4:                                               ; preds = %b3
  %v16 = phi i16 [ %v12, %b3 ]
  %v17 = phi i32 [ %v11, %b3 ]
  %v18 = phi i32 [ %v10, %b3 ]
  store i16 %v16, i16* %v1, align 2, !tbaa !4
  %v19 = icmp slt i32 %v18, 1
  br i1 %v19, label %b5, label %b6

b5:                                               ; preds = %b4
  %v20 = call signext i16 @f2(float* %v0, i32 %v17) #1
  br label %b6

b6:                                               ; preds = %b5, %b4, %b1
  %v21 = bitcast float* %v0 to i16*
  %v22 = getelementptr inbounds i16, i16* %v21, i32 1
  %v23 = load i16, i16* %v22, align 2, !tbaa !6
  %v24 = icmp slt i16 %v23, 0
  %v25 = load float, float* %v0, align 4, !tbaa !0
  br i1 %v24, label %b7, label %b8

b7:                                               ; preds = %b6
  %v26 = fsub float -0.000000e+00, %v25
  store float %v26, float* %v0, align 4, !tbaa !0
  br label %b8

b8:                                               ; preds = %b7, %b6
  %v27 = phi float [ %v26, %b7 ], [ %v25, %b6 ]
  %v28 = phi i1 [ true, %b7 ], [ false, %b6 ]
  %v29 = fmul float %v27, 0x3FCF3482C0000000
  %v30 = fadd float %v29, 0x3FEEA88260000000
  %v31 = fmul float %v27, %v30
  %v32 = fadd float %v31, 0x3FB43419E0000000
  %v33 = fadd float %v27, 0x3FD1E54B40000000
  %v34 = fdiv float %v32, %v33
  store float %v34, float* %v2, align 4, !tbaa !0
  %v35 = fmul float %v27, 1.500000e+00
  %v36 = fmul float %v34, %v34
  %v37 = fmul float %v27, 5.000000e-01
  %v38 = fdiv float %v37, %v34
  %v39 = fadd float %v36, %v38
  %v40 = fdiv float %v35, %v39
  %v41 = fadd float %v34, %v40
  %v42 = fmul float %v41, 5.000000e-01
  br i1 %v28, label %b9, label %b10

b9:                                               ; preds = %b8
  %v43 = fsub float -0.000000e+00, %v42
  br label %b10

b10:                                              ; preds = %b9, %b8
  %v44 = phi float [ %v43, %b9 ], [ %v42, %b8 ]
  store float %v44, float* %v2, align 4, !tbaa !0
  %v45 = load i16, i16* %v1, align 2, !tbaa !4
  %v46 = sext i16 %v45 to i32
  %v47 = sdiv i32 %v46, 3
  %v48 = call signext i16 @f2(float* %v2, i32 %v47) #1
  br label %b11

b11:                                              ; preds = %b10, %b0
  %v49 = phi float* [ %v2, %b10 ], [ %v0, %b0 ]
  %v50 = load float, float* %v49, align 4
  ret float %v50
}

declare signext i16 @f1(i16*, float*) #1

declare signext i16 @f2(float*, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
attributes #1 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"float", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"short", !2, i64 0}
!6 = !{!2, !2, i64 0}
