; RUN: llc -march=hexagon -O2 -debug-only=pipeliner -hexagon-initial-cfg-cleanup=0 < %s -o - 2>&1 > /dev/null | FileCheck %s
; REQUIRES: asserts

; Test that the phi in the first epilog block is getter the correct
; value from the kernel block. In this bug, the phi was using the value
; defined in the loop instead of the Phi valued defined in the kernel.
; We need to use the kernel's phi value (if the Phi in the kernel is the
; last definition).

; CHECK: New block
; CHECK: %[[REG:([0-9]+)]]:intregs = PHI %{{.*}}, %[[REG1:([0-9]+)]]
; CHECK: %[[REG1]]:intregs = nuw A2_addi
; CHECK: epilog:
; CHECK: %{{[0-9]+}}:intregs = PHI %{{.*}}, %[[REG]]

define void @f0(i32 %a0, i32 %a1) #0 {
b0:
  %v0 = icmp sgt i32 %a0, 64
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i32 [ %a0, %b1 ], [ %v13, %b2 ]
  %v2 = phi <16 x i32>* [ null, %b1 ], [ %v3, %b2 ]
  %v3 = getelementptr inbounds <16 x i32>, <16 x i32>* %v2, i32 1
  %v4 = load <16 x i32>, <16 x i32>* %v2, align 64
  %v5 = load <16 x i32>, <16 x i32>* undef, align 64
  %v6 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v5, <16 x i32> undef, i32 1)
  %v7 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> undef, <16 x i32> %v6)
  %v8 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> undef, <32 x i32> %v7, i32 undef)
  %v9 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v8, <32 x i32> undef, i32 undef)
  %v10 = tail call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> %v9, <16 x i32> zeroinitializer, i32 undef)
  %v11 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v10)
  %v12 = tail call <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32> undef, <16 x i32> %v11, i32 %a1)
  store <16 x i32> %v12, <16 x i32>* null, align 64
  %v13 = add nsw i32 %v1, -64
  %v14 = icmp sgt i32 %v13, 64
  br i1 %v14, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32>, <16 x i32>, i32) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32>, <32 x i32>, i32) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32>, <32 x i32>, i32) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32>, <16 x i32>, i32) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32>, <16 x i32>, i32) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #0

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
