; RUN: llc -fp-contract=fast -O3 -mtriple=hexagon < %s
; REQUIRES: asserts

; Test that the pipeliner doesn't ICE due because the PHI generation
; code in the epilog does not attempt to reuse an existing PHI.
; Similar test case as swp-epilog-reuse.ll but with a couple of
; differences.

; Function Attrs: nounwind
define void @f0(ptr noalias %a0, ptr noalias %a1) #0 {
b0:
  %v0 = getelementptr inbounds float, ptr %a1, i32 2
  br i1 undef, label %b1, label %b6

b1:                                               ; preds = %b5, %b0
  %v1 = phi ptr [ undef, %b5 ], [ %v0, %b0 ]
  %v2 = phi ptr [ %v32, %b5 ], [ undef, %b0 ]
  %v3 = getelementptr inbounds float, ptr %a0, i32 undef
  %v4 = getelementptr inbounds float, ptr %v1, i32 1
  br i1 undef, label %b2, label %b5

b2:                                               ; preds = %b1
  %v5 = getelementptr float, ptr %v3, i32 1
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v6 = phi ptr [ %v5, %b2 ], [ %v20, %b3 ]
  %v7 = phi float [ %v19, %b3 ], [ undef, %b2 ]
  %v8 = phi float [ %v7, %b3 ], [ undef, %b2 ]
  %v9 = phi ptr [ %v15, %b3 ], [ %v4, %b2 ]
  %v11 = fadd float undef, 0.000000e+00
  %v12 = fadd float undef, %v11
  %v13 = fadd float %v7, %v12
  %v14 = fmul float %v13, 3.906250e-03
  %v15 = getelementptr inbounds float, ptr %v9, i32 1
  store float %v14, ptr %v9, align 4, !tbaa !0
  %v16 = getelementptr i8, ptr %v6, i32 undef
  %v18 = load float, ptr %v16, align 4, !tbaa !0
  %v19 = fadd float %v18, undef
  %v20 = getelementptr float, ptr %v6, i32 2
  %v21 = icmp ult ptr %v15, %v2
  br i1 %v21, label %b3, label %b4

b4:                                               ; preds = %b3
  %v22 = getelementptr float, ptr %v4, i32 undef
  br label %b5

b5:                                               ; preds = %b4, %b1
  %v23 = phi ptr [ %v4, %b1 ], [ %v22, %b4 ]
  %v24 = phi float [ undef, %b1 ], [ %v8, %b4 ]
  %v25 = fadd float %v24, undef
  %v26 = fadd float %v25, undef
  %v27 = fadd float undef, %v26
  %v28 = fadd float undef, %v27
  %v29 = fpext float %v28 to double
  %v30 = fmul double %v29, 0x3F7111112119E8FB
  %v31 = fptrunc double %v30 to float
  store float %v31, ptr %v23, align 4, !tbaa !0
  %v32 = getelementptr inbounds float, ptr %v2, i32 undef
  br i1 undef, label %b1, label %b6

b6:                                               ; preds = %b5, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
