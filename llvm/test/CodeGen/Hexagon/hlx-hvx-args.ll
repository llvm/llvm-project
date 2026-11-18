; Verify that HVX args before HLX args in the signature land in $v/$l registers
; in declaration order, and that interleaved (hvx, hlx, hvx, hlx) signatures
; also route each type to its own register bank in order.
;
; para_hvx_hlx(HVX h0, HVX h1, HLX v0, HLX v1)            -> liveins $v0,$v1,$l0,$l1
; para_interleave_hvx_hlx(HVX h0, HLX v0, HVX h1, HLX v1) -> liveins $v0,$l0,$v1,$l1

; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx,+hvx-length128b,+hvxv85 -start-before=hexagon-isel -stop-after=hexagon-isel -verify-machineinstrs -o - %s | FileCheck %s

@vtcm_base = dso_local local_unnamed_addr global ptr null, align 4

; CHECK-LABEL: name: para_hvx_hlx
; CHECK:       liveins: $v0, $v1, $l0, $l1
define dso_local inreg <32 x i32> @para_hvx_hlx(<32 x i32> %h0, <32 x i32> %h1, <1024 x i16> %v0, <1024 x i16> %v1) local_unnamed_addr #0 {
entry:
  %0 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %v0, <1024 x i16> %v1)
  %1 = load ptr, ptr @vtcm_base, align 4
  store <1024 x i16> %0, ptr %1, align 2048
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vaddb.128B(<32 x i32> %h0, <32 x i32> %h1)
  ret <32 x i32> %2
}

; CHECK-LABEL: name: para_interleave_hvx_hlx
; CHECK:       liveins: $v0, $l0, $v1, $l1
define dso_local inreg <32 x i32> @para_interleave_hvx_hlx(<32 x i32> %h0, <1024 x i16> %v0, <32 x i32> %h1, <1024 x i16> %v1) local_unnamed_addr #0 {
entry:
  %0 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %v0, <1024 x i16> %v1)
  %1 = load ptr, ptr @vtcm_base, align 4
  store <1024 x i16> %0, ptr %1, align 2048
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vaddb.128B(<32 x i32> %h0, <32 x i32> %h1)
  ret <32 x i32> %2
}

; CHECK-LABEL: name: main
; para_hvx_hlx(v0=1, v1=2, l0=0, l1=1)
; CHECK-DAG:   %[[L0:[0-9]+]]:hlxlr = LX_assign_u 0
; CHECK-DAG:   %[[L1:[0-9]+]]:hlxlr = LX_assign_u 1
; CHECK-DAG:   %[[RI1:[0-9]+]]:intregs = A2_tfrsi 1
; CHECK-DAG:   %[[H0:[0-9]+]]:hvxvr = V6_lvsplatb {{.*}}%[[RI1]]
; CHECK-DAG:   %[[RI2:[0-9]+]]:intregs = A2_tfrsi 2
; CHECK-DAG:   %[[H1:[0-9]+]]:hvxvr = V6_lvsplatb {{.*}}%[[RI2]]
; CHECK-DAG:   $v0 = COPY %[[H0]]
; CHECK-DAG:   $v1 = COPY %[[H1]]
; CHECK-DAG:   $l0 = COPY %[[L0]]
; CHECK-DAG:   $l1 = COPY %[[L1]]
; CHECK:       J2_call @para_hvx_hlx, hexagoncsr_hlx,
; CHECK-SAME:  implicit $v0, implicit $v1, implicit $l0, implicit $l1
; para_interleave_hvx_hlx(v0=3, l0=2, v1=4, l1=3)
; CHECK-DAG:   %[[L2:[0-9]+]]:hlxlr = LX_assign_u 2
; CHECK-DAG:   %[[L3:[0-9]+]]:hlxlr = LX_assign_u 3
; CHECK-DAG:   %[[RI3:[0-9]+]]:intregs = A2_tfrsi 3
; CHECK-DAG:   %[[H2:[0-9]+]]:hvxvr = V6_lvsplatb {{.*}}%[[RI3]]
; CHECK-DAG:   %[[RI4:[0-9]+]]:intregs = A2_tfrsi 4
; CHECK-DAG:   %[[H3:[0-9]+]]:hvxvr = V6_lvsplatb {{.*}}%[[RI4]]
; CHECK-DAG:   $v0 = COPY %[[H2]]
; CHECK-DAG:   $l0 = COPY %[[L2]]
; CHECK-DAG:   $v1 = COPY %[[H3]]
; CHECK-DAG:   $l1 = COPY %[[L3]]
; CHECK:       J2_call @para_interleave_hvx_hlx, hexagoncsr_hlx,
; CHECK-SAME:  implicit $v0, implicit $l0, implicit $v1, implicit $l1
define dso_local i32 @main() local_unnamed_addr #2 {
entry:
  %0 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 0)
  %1 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 1)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatb.128B(i32 1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatb.128B(i32 2)
  %call = tail call inreg <32 x i32> @para_hvx_hlx(<32 x i32> %2, <32 x i32> %3, <1024 x i16> %0, <1024 x i16> %1)
  %4 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 2)
  %5 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 3)
  %6 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatb.128B(i32 3)
  %7 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatb.128B(i32 4)
  %call2 = tail call inreg <32 x i32> @para_interleave_hvx_hlx(<32 x i32> %6, <1024 x i16> %4, <32 x i32> %7, <1024 x i16> %5)
  ret i32 0
}

declare <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16>, <1024 x i16>) #1
declare <32 x i32> @llvm.hexagon.V6.vaddb.128B(<32 x i32>, <32 x i32>) #1
declare <1024 x i16> @llvm.hexagon.LX.assign.u(i32 immarg) #1
declare <32 x i32> @llvm.hexagon.V6.lvsplatb.128B(i32) #1

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nofree nounwind }
