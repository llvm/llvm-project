; RUN: llc -march=hexagon -O2 -enable-pipeliner=false < %s | FileCheck %s
; RUN: llc -march=hexagon -O2 -debug-only=pipeliner < %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-SWP
; REQUIRES: asserts

; CHECK: {
; CHECK-DAG: v{{[0-9]*}} = vmem(r{{[0-9]*}}++#1)
; CHECK-DAG: vmem(r{{[0-9]*}}++#1) = v{{[0-9]*}}.new
; CHECK: }{{[ \t]*}}:endloop0

; CHECK-SWP: Schedule Found? 1
; CHECK-SWP: {
; CHECK-DAG-SWP: v{{[0-9]*}}.cur = vmem(r{{[0-9]*}}++#1)
; CHECK-DAG-SWP: vmem(r{{[0-9]*}}++#1) = v{{[0-9]*}}.new
; CHECK-SWP: }{{[ \t]*}}:endloop0

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0(ptr nocapture readonly %a0, i32 %a1, i32 %a2, ptr nocapture %a3) #0 {
b0:
  %v0 = mul i32 %a2, -2
  %v1 = add i32 %v0, 64
  %v2 = tail call <16 x i32> @llvm.hexagon.V6.vsubw(<16 x i32> undef, <16 x i32> undef)
  %v4 = sdiv i32 %a1, 32
  %v5 = icmp sgt i32 %a1, 31
  br i1 %v5, label %b1, label %b4

b1:                                               ; preds = %b0
  %v7 = icmp sgt i32 %a1, 63
  %v8 = mul i32 %v4, 32
  %v9 = select i1 %v7, i32 %v8, i32 32
  %v10 = getelementptr i16, ptr %a3, i32 %v9
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v11 = phi i32 [ 0, %b1 ], [ %v19, %b2 ]
  %v12 = phi <16 x i32> [ %v2, %b1 ], [ %v16, %b2 ]
  %v13 = phi ptr [ %a3, %b1 ], [ %v18, %b2 ]
  %v14 = phi ptr [ %a0, %b1 ], [ %v15, %b2 ]
  %v15 = getelementptr inbounds <16 x i32>, ptr %v14, i32 1
  %v16 = load <16 x i32>, ptr %v14, align 64, !tbaa !0
  %v17 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v16, <16 x i32> %v12, i32 %v1)
  %v18 = getelementptr inbounds <16 x i32>, ptr %v13, i32 1
  store <16 x i32> %v17, ptr %v13, align 64, !tbaa !0
  %v19 = add nsw i32 %v11, 1
  %v20 = icmp slt i32 %v19, %v4
  br i1 %v20, label %b2, label %b3

b3:                                               ; preds = %b2
  br label %b4

b4:                                               ; preds = %b3, %b0
  %v22 = phi <16 x i32> [ %v16, %b3 ], [ %v2, %b0 ]
  %v23 = phi ptr [ %v10, %b3 ], [ %a3, %b0 ]
  %v24 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v2, <16 x i32> %v22, i32 %v1)
  store <16 x i32> %v24, ptr %v23, align 64, !tbaa !0
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubw(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32>, <16 x i32>, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
