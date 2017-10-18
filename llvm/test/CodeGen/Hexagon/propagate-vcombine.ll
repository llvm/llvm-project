; RUN: llc -march=hexagon < %s | FileCheck %s

@v0 = global <16 x i32> zeroinitializer, align 64
@v1 = global <16 x i32> zeroinitializer, align 64

; CHECK-LABEL: danny:
; CHECK-NOT: vcombine

define void @danny() #0 {
  %t0 = load <16 x i32>, <16 x i32>* @v0, align 64
  %t1 = load <16 x i32>, <16 x i32>* @v1, align 64
  %t2 = call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %t0, <16 x i32> %t1)
  %t3 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %t2)
  %t4 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %t2)
  store <16 x i32> %t3, <16 x i32>* @v0, align 64
  store <16 x i32> %t4, <16 x i32>* @v1, align 64
  ret void
}

@w0 = global <32 x i32> zeroinitializer, align 128
@w1 = global <32 x i32> zeroinitializer, align 128

; CHECK-LABEL: sammy:
; CHECK-NOT: vcombine

define void @sammy() #1 {
  %t0 = load <32 x i32>, <32 x i32>* @w0, align 128
  %t1 = load <32 x i32>, <32 x i32>* @w1, align 128
  %t2 = call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %t0, <32 x i32> %t1)
  %t3 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %t2)
  %t4 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %t2)
  store <32 x i32> %t3, <32 x i32>* @w0, align 128
  store <32 x i32> %t4, <32 x i32>* @w1, align 128
  ret void
}

declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #2
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #2
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #2

declare <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32>, <32 x i32>) #3
declare <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32>) #3
declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #3

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
attributes #2 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #3 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
