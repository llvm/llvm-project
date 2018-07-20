; RUN: llc -fp-contract=fast -O3 -march=hexagon < %s
; REQUIRES: asserts

; Test that the pipeliner doesn't ICE due because the PHI generation
; code in the epilog does not attempt to reuse an existing PHI.
; Similar test case as swp-epilog-reuse.ll but with a couple of
; differences.

; Function Attrs: nounwind
define void @f0(float* noalias %a0, float* noalias %a1) #0 {
b0:
  %v0 = getelementptr inbounds float, float* %a1, i32 2
  br i1 undef, label %b1, label %b6

b1:                                               ; preds = %b5, %b0
  %v1 = phi float* [ undef, %b5 ], [ %v0, %b0 ]
  %v2 = phi float* [ %v32, %b5 ], [ undef, %b0 ]
  %v3 = getelementptr inbounds float, float* %a0, i32 undef
  %v4 = getelementptr inbounds float, float* %v1, i32 1
  br i1 undef, label %b2, label %b5

b2:                                               ; preds = %b1
  %v5 = getelementptr float, float* %v3, i32 1
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v6 = phi float* [ %v5, %b2 ], [ %v20, %b3 ]
  %v7 = phi float [ %v19, %b3 ], [ undef, %b2 ]
  %v8 = phi float [ %v7, %b3 ], [ undef, %b2 ]
  %v9 = phi float* [ %v15, %b3 ], [ %v4, %b2 ]
  %v10 = bitcast float* %v6 to i8*
  %v11 = fadd float undef, 0.000000e+00
  %v12 = fadd float undef, %v11
  %v13 = fadd float %v7, %v12
  %v14 = fmul float %v13, 3.906250e-03
  %v15 = getelementptr inbounds float, float* %v9, i32 1
  store float %v14, float* %v9, align 4, !tbaa !0
  %v16 = getelementptr i8, i8* %v10, i32 undef
  %v17 = bitcast i8* %v16 to float*
  %v18 = load float, float* %v17, align 4, !tbaa !0
  %v19 = fadd float %v18, undef
  %v20 = getelementptr float, float* %v6, i32 2
  %v21 = icmp ult float* %v15, %v2
  br i1 %v21, label %b3, label %b4

b4:                                               ; preds = %b3
  %v22 = getelementptr float, float* %v4, i32 undef
  br label %b5

b5:                                               ; preds = %b4, %b1
  %v23 = phi float* [ %v4, %b1 ], [ %v22, %b4 ]
  %v24 = phi float [ undef, %b1 ], [ %v8, %b4 ]
  %v25 = fadd float %v24, undef
  %v26 = fadd float %v25, undef
  %v27 = fadd float undef, %v26
  %v28 = fadd float undef, %v27
  %v29 = fpext float %v28 to double
  %v30 = fmul double %v29, 0x3F7111112119E8FB
  %v31 = fptrunc double %v30 to float
  store float %v31, float* %v23, align 4, !tbaa !0
  %v32 = getelementptr inbounds float, float* %v2, i32 undef
  br i1 undef, label %b1, label %b6

b6:                                               ; preds = %b5, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
