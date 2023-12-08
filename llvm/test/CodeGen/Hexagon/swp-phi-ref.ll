; RUN: llc -march=hexagon -enable-pipeliner -enable-bsb-sched=0 -join-liveintervals=false < %s -pipeliner-experimental-cg=true | FileCheck %s

; XFAIL: *
; This test is failing after post-ra machine sinking.

; Test that we generate the correct Phi values when there is a Phi that
; references another Phi. We need to examine the other Phi to get the
; correct value. We need to do this even if we haven't generated the
; kernel code for the other Phi yet.

; CHECK: v[[REG0:[0-9]+]] = v[[REG1:[0-9]+]]
; CHECK: loop0
; Check for copy REG0 = REG1 (via vcombine):
; CHECK: v{{[0-9]+}}:[[REG0]] = vcombine(v{{[0-9]+}},v[[REG1]])
; CHECK: endloop0

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ %v7, %b1 ], [ 0, %b0 ]
  %v1 = phi <16 x i32> [ %v4, %b1 ], [ undef, %b0 ]
  %v2 = phi <16 x i32> [ %v1, %b1 ], [ undef, %b0 ]
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v1, <16 x i32> %v2, i32 62)
  %v4 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> undef, <16 x i32> undef)
  %v5 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v4, <16 x i32> %v1, i32 2)
  %v6 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffh(<16 x i32> %v3, <16 x i32> %v5)
  store <16 x i32> %v6, ptr null, align 64
  %v7 = add nsw i32 %v0, 1
  %v8 = icmp slt i32 %v7, undef
  br i1 %v8, label %b1, label %b2

b2:                                               ; preds = %b1, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vabsdiffh(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
