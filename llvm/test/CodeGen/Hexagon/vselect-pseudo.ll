; RUN: llc -mtriple=hexagon -mattr="+hvxv60,+hvx-length64b" < %s
; REQUIRES: asserts

target triple = "hexagon"

; Function Attrs: nounwind
define void @fred() #0 {
entry:
  br label %for.body9.us

for.body9.us:
  %cmp10.us = icmp eq i32 0, undef
  %.h63h32.2.us = select i1 %cmp10.us, <16 x i32> zeroinitializer, <16 x i32> undef
  %0 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %.h63h32.2.us, <16 x i32> undef, i32 2)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vswap(<64 x i1> undef, <16 x i32> undef, <16 x i32> %0)
  %2 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32> undef, <16 x i32> %2, i32 62)
  %4 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %3)
  store <16 x i32> %4, ptr undef, align 64
  br i1 undef, label %for.body9.us, label %for.body43.us.preheader

for.body43.us.preheader:                          ; preds = %for.body9.us
  ret void
}

declare <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32>, <16 x i32>, i32) #1
declare <32 x i32> @llvm.hexagon.V6.vswap(<64 x i1>, <16 x i32>, <16 x i32>) #1
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #1
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32>, <16 x i32>, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
