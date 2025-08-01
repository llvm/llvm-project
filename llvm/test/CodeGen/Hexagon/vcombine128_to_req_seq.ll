; RUN: llc -mtriple=hexagon -O2 < %s | FileCheck %s

; Test that we convert 128B vcombine instructions to REG_SEQUENCE instructions.

; CHECK-LABEL: f0:
; CHECK-NOT: vcombine
define void @f0(ptr nocapture readonly %a0, ptr nocapture readonly %a1, i32 %a2, ptr nocapture %a3, i32 %a4, i32 %a5) #0 {
b0:
  %v1 = load i64, ptr %a1, align 8
  %v2 = shl i64 %v1, 8
  %v3 = trunc i64 %v2 to i32
  %v4 = trunc i64 %v1 to i32
  %v5 = and i32 %v4, 16777215
  %v7 = load <32 x i32>, ptr %a0, align 128
  %v8 = getelementptr inbounds i8, ptr %a0, i32 32
  %v10 = load <32 x i32>, ptr %v8, align 128
  %v11 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %v10, <32 x i32> %v7)
  %v12 = tail call <64 x i32> @llvm.hexagon.V6.vrmpybusi.128B(<64 x i32> %v11, i32 %v5, i32 0)
  %v13 = tail call <64 x i32> @llvm.hexagon.V6.vrmpybusi.128B(<64 x i32> %v11, i32 %v3, i32 0)
  %v14 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v12)
  %v15 = tail call <32 x i32> @llvm.hexagon.V6.vasrwuhsat.128B(<32 x i32> %v14, <32 x i32> %v14, i32 %a2)
  %v16 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v13)
  %v17 = tail call <32 x i32> @llvm.hexagon.V6.vasrwuhsat.128B(<32 x i32> %v16, <32 x i32> %v16, i32 %a2)
  %v18 = getelementptr inbounds i8, ptr %a3, i32 32
  store <32 x i32> %v15, ptr %v18, align 128
  store <32 x i32> %v17, ptr %a3, align 128
  ret void
}

; CHECK-LABEL: f1:
; CHECK-NOT: vcombine
define void @f1() #0 {
b0:
  br i1 undef, label %b1, label %b3

b1:                                               ; preds = %b1, %b0
  %v0 = phi <64 x i32> [ %v6, %b1 ], [ undef, %b0 ]
  %v1 = tail call <64 x i32> @llvm.hexagon.V6.vmpybus.acc.128B(<64 x i32> %v0, <32 x i32> undef, i32 16843009)
  %v2 = tail call <64 x i32> @llvm.hexagon.V6.vmpabus.acc.128B(<64 x i32> %v1, <64 x i32> undef, i32 16843009)
  %v3 = tail call <64 x i32> @llvm.hexagon.V6.vmpabus.acc.128B(<64 x i32> %v2, <64 x i32> undef, i32 16843009)
  %v4 = tail call <64 x i32> @llvm.hexagon.V6.vmpabus.acc.128B(<64 x i32> %v3, <64 x i32> undef, i32 16843009)
  %v5 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> undef, <32 x i32> undef)
  %v6 = tail call <64 x i32> @llvm.hexagon.V6.vmpabus.acc.128B(<64 x i32> %v4, <64 x i32> %v5, i32 16843009)
  br i1 false, label %b2, label %b1

b2:                                               ; preds = %b1
  %v7 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v6)
  unreachable

b3:                                               ; preds = %b0
  ret void
}

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vrmpybusi.128B(<64 x i32>, i32, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vasrwuhsat.128B(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32>) #1

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vmpybus.acc.128B(<64 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vmpabus.acc.128B(<64 x i32>, <64 x i32>, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
attributes #1 = { nounwind readnone }
