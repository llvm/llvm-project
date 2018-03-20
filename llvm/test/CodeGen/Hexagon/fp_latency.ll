; RUN: llc -O2 -march=hexagon -fp-contract=fast -pipeliner-prune-loop-carried=false < %s | FileCheck %s

; Test that there is 1 packet between the FP result and its use.

; CHECK: loop0([[LOOP0:.LBB[0-9_]+]],
; CHECK: [[LOOP0]]
; CHECK: [[REG0:(r[0-9]+)]] += sfmpy(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: }
; CHECK: }
; CHECK: r{{[0-9]+}} = {{.*}}[[REG0]]

; Function Attrs: nounwind readnone
define void @f0(i32 %a0, i32 %a1) #0 {
b0:
  %v0 = alloca [1000 x float], align 64
  %v1 = alloca [1000 x float], align 64
  %v2 = alloca [1000 x float], align 64
  %v3 = alloca [1000 x float], align 64
  %v4 = bitcast [1000 x float]* %v0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4000, i8* %v4) #2
  %v5 = bitcast [1000 x float]* %v1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4000, i8* %v5) #2
  %v6 = bitcast [1000 x float]* %v2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4000, i8* %v6) #2
  %v7 = bitcast [1000 x float]* %v3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4000, i8* %v7) #2
  %v8 = icmp sgt i32 %a1, 0
  %v9 = add i32 %a1, -1
  %v10 = getelementptr [1000 x float], [1000 x float]* %v3, i32 0, i32 0
  br label %b1

b1:                                               ; preds = %b3, %b0
  %v11 = phi i32 [ 0, %b0 ], [ %v34, %b3 ]
  br i1 %v8, label %b2, label %b3

b2:                                               ; preds = %b2, %b1
  %v12 = phi float* [ %v33, %b2 ], [ %v10, %b1 ]
  %v13 = phi i32 [ %v31, %b2 ], [ 0, %b1 ]
  %v14 = mul nsw i32 %v13, %a1
  %v15 = add nsw i32 %v14, %v11
  %v16 = getelementptr inbounds [1000 x float], [1000 x float]* %v1, i32 0, i32 %v15
  %v17 = load float, float* %v16, align 4, !tbaa !0
  %v18 = fmul float %v17, undef
  %v19 = mul nsw i32 %v13, 25
  %v20 = add nsw i32 %v19, %v11
  %v21 = getelementptr inbounds [1000 x float], [1000 x float]* %v2, i32 0, i32 %v20
  %v22 = load float, float* %v21, align 4, !tbaa !0
  %v23 = fmul float %v22, undef
  %v24 = fadd float %v18, %v23
  %v25 = load float, float* %v12, align 4, !tbaa !0
  %v26 = fmul float %v25, undef
  %v27 = fadd float %v24, %v26
  %v28 = getelementptr inbounds [1000 x float], [1000 x float]* %v0, i32 0, i32 %v20
  %v29 = load float, float* %v28, align 4, !tbaa !0
  %v30 = fadd float %v29, %v27
  store float %v30, float* %v28, align 4, !tbaa !0
  %v31 = add nuw nsw i32 %v13, 1
  %v32 = icmp eq i32 %v13, %v9
  %v33 = getelementptr float, float* %v12, i32 1
  br i1 %v32, label %b3, label %b2

b3:                                               ; preds = %b2, %b1
  %v34 = add nuw nsw i32 %v11, 1
  %v35 = icmp eq i32 %v34, 25
  br i1 %v35, label %b4, label %b1

b4:                                               ; preds = %b3
  call void @llvm.lifetime.end.p0i8(i64 4000, i8* %v7) #2
  call void @llvm.lifetime.end.p0i8(i64 4000, i8* %v6) #2
  call void @llvm.lifetime.end.p0i8(i64 4000, i8* %v5) #2
  call void @llvm.lifetime.end.p0i8(i64 4000, i8* %v4) #2
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"float", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
