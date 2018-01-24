; RUN: llc -march=hexagon -verify-machineinstrs < %s | FileCheck %s
;
; Check that this does not crash.

target triple = "hexagon"

; CHECK-LABEL: danny:
; CHECK-DAG: if ([[PREG:p[0-3]]]) [[VREG:v[0-9]+]]
; CHECK-DAG: if (![[PREG]]) [[VREG]]
define void @danny(i32 %a0) local_unnamed_addr #0 {
b0:
  %v1 = icmp eq i32 0, %a0
  %v2 = select i1 %v1, <16 x i32> zeroinitializer, <16 x i32> undef
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v2, <16 x i32> zeroinitializer, i32 2)
  %v4 = tail call <32 x i32> @llvm.hexagon.V6.vswap(<512 x i1> undef, <16 x i32> undef, <16 x i32> %v3)
  %v5 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v4)
  %v6 = tail call <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32> undef, <16 x i32> %v5, i32 62)
  %v7 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v6)
  store <16 x i32> %v7, <16 x i32>* undef, align 64
  unreachable
}

declare <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32>, <16 x i32>, i32) #2
declare <32 x i32> @llvm.hexagon.V6.vswap(<512 x i1>, <16 x i32>, <16 x i32>) #2
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #2
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #2
declare <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32>, <16 x i32>, i32) #2

; CHECK-LABEL: sammy:
; CHECK-DAG: if ([[PREG:p[0-3]]]) [[VREG:v[0-9]+]]
; CHECK-DAG: if (![[PREG]]) [[VREG]]
define void @sammy(i32 %a0) local_unnamed_addr #1 {
b0:
  %v1 = icmp eq i32 0, %a0
  %v2 = select i1 %v1, <32 x i32> zeroinitializer, <32 x i32> undef
  %v3 = tail call <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32> %v2, <32 x i32> zeroinitializer, i32 2)
  %v4 = tail call <64 x i32> @llvm.hexagon.V6.vswap.128B(<1024 x i1> undef, <32 x i32> undef, <32 x i32> %v3)
  %v5 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v4)
  %v6 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> undef, <32 x i32> %v5, i32 62)
  %v7 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v6)
  store <32 x i32> %v7, <32 x i32>* undef, align 128
  unreachable
}

declare <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32>, <32 x i32>, i32) #2
declare <64 x i32> @llvm.hexagon.V6.vswap.128B(<1024 x i1>, <32 x i32>, <32 x i32>) #2
declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #2
declare <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32>) #2
declare <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32>, <32 x i32>, i32) #2

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
attributes #2 = { nounwind readnone }
