; RUN: llc -march=hexagon -hexagon-initial-cfg-cleanup=0 < %s | FileCheck %s
; CHECK: = vmem(r{{[0-9]+}}++#1)

target triple = "hexagon-unknown--elf"

declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32>, <32 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vzb.128B(<32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vsathub.128B(<32 x i32>, <32 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vaddh.dv.128B(<64 x i32>, <64 x i32>) #0
declare <64 x i32> @llvm.hexagon.V6.vadduhsat.dv.128B(<64 x i32>, <64 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vabsdiffuh.128B(<32 x i32>, <32 x i32>) #0

define void @f0(ptr %a0, ptr %a1) #1 {
b0:
  br label %b1

b1:                                               ; preds = %b2, %b1
  %v0 = phi <128 x i8> [ %v7, %b1 ], [ undef, %b0 ]
  %v1 = phi i32 [ %v19, %b1 ], [ 0, %b0 ]
  %v2 = add nsw i32 %v1, undef
  %v3 = shl i32 %v2, 7
  %v4 = add nsw i32 %v3, 128
  %v5 = getelementptr inbounds i8, ptr %a0, i32 %v4
  %v7 = load <128 x i8>, ptr %v5, align 128
  %v8 = bitcast <128 x i8> %v0 to <32 x i32>
  %v9 = tail call <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32> undef, <32 x i32> %v8, i32 1)
  %v10 = tail call <64 x i32> @llvm.hexagon.V6.vzb.128B(<32 x i32> %v9) #1
  %v11 = tail call <64 x i32> @llvm.hexagon.V6.vadduhsat.dv.128B(<64 x i32> undef, <64 x i32> %v10) #1
  %v12 = tail call <64 x i32> @llvm.hexagon.V6.vadduhsat.dv.128B(<64 x i32> %v11, <64 x i32> undef) #1
  %v13 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v12)
  %v14 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffuh.128B(<32 x i32> undef, <32 x i32> %v13) #1
  %v15 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %v14, <32 x i32> undef)
  %v16 = tail call <64 x i32> @llvm.hexagon.V6.vaddh.dv.128B(<64 x i32> undef, <64 x i32> %v15) #1
  %v17 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v16) #1
  %v18 = tail call <32 x i32> @llvm.hexagon.V6.vsathub.128B(<32 x i32> %v17, <32 x i32> undef) #1
  store <32 x i32> %v18, ptr %a1, align 128
  %v19 = add nuw nsw i32 %v1, 1
  br label %b1
}

declare <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32>, <32 x i32>, i32) #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
