; Verify that HVX args before HLX_VectorPair args in the signature land in $v/$ll registers
; in declaration order, and that interleaved (hvx, pair, hvx, pair) signatures
; also route each type to its own register bank in order.
;
; para_pair_hvx_hlx_pair(HVX h0, HVX h1, HLX_VectorPair p0, HLX_VectorPair p1)
;   -> liveins $v0, $v1, $ll0, $ll1
; para_interleave_hvx_hlx_pair(HVX h0, HLX_VectorPair p0, HVX h1, HLX_VectorPair p1)
;   -> liveins $v0, $ll0, $v1, $ll1

; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx,+hvx-length128b,+hvxv85 -start-before=hexagon-isel -stop-after=hexagon-isel -verify-machineinstrs -o - %s | FileCheck %s

@vtcm_base = dso_local local_unnamed_addr global ptr null, align 4

; CHECK-LABEL: name: para_pair_hvx_hlx_pair
; CHECK:       liveins: $v0, $v1, $ll0, $ll1
define dso_local inreg <32 x i32> @para_pair_hvx_hlx_pair(<32 x i32> %h0, <32 x i32> %h1, <1024 x i32> %p0, <1024 x i32> %p1) local_unnamed_addr #0 {
entry:
  %0 = tail call <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32> %p0, <1024 x i32> %p1)
  %1 = tail call <1024 x i16> @llvm.hexagon.LX.hi(<1024 x i32> %0)
  %2 = tail call <1024 x i16> @llvm.hexagon.LX.lo(<1024 x i32> %0)
  %3 = load ptr, ptr @vtcm_base, align 4, !tbaa !6
  store <1024 x i16> %1, ptr %3, align 2048, !tbaa !9
  %cgep = getelementptr inbounds i8, ptr %3, i32 2048
  store <1024 x i16> %2, ptr %cgep, align 2048, !tbaa !9
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vaddb.128B(<32 x i32> %h0, <32 x i32> %h1)
  ret <32 x i32> %4
}

; CHECK-LABEL: name: para_interleave_hvx_hlx_pair
; CHECK:       liveins: $v0, $ll0, $v1, $ll1
define dso_local inreg <32 x i32> @para_interleave_hvx_hlx_pair(<32 x i32> %h0, <1024 x i32> %p0, <32 x i32> %h1, <1024 x i32> %p1) local_unnamed_addr #0 {
entry:
  %0 = tail call <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32> %p0, <1024 x i32> %p1)
  %1 = tail call <1024 x i16> @llvm.hexagon.LX.hi(<1024 x i32> %0)
  %2 = tail call <1024 x i16> @llvm.hexagon.LX.lo(<1024 x i32> %0)
  %3 = load ptr, ptr @vtcm_base, align 4, !tbaa !6
  store <1024 x i16> %1, ptr %3, align 2048, !tbaa !9
  %cgep = getelementptr inbounds i8, ptr %3, i32 2048
  store <1024 x i16> %2, ptr %cgep, align 2048, !tbaa !9
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vaddb.128B(<32 x i32> %h0, <32 x i32> %h1)
  ret <32 x i32> %4
}

; CHECK-LABEL: name: main
; para_pair_hvx_hlx_pair(v0=1, v1=2, ll0=combine(0,1), ll1=combine(2,3))
; CHECK-DAG:   %[[L0:[0-9]+]]:hlxlr = LX_assign_u 0
; CHECK-DAG:   %[[L1:[0-9]+]]:hlxlr = LX_assign_u 1
; CHECK-DAG:   %[[L2:[0-9]+]]:hlxlr = LX_assign_u 2
; CHECK-DAG:   %[[L3:[0-9]+]]:hlxlr = LX_assign_u 3
; CHECK-DAG:   %[[P0:[0-9]+]]:hlxllr = LX_combine_w {{.*}}%[[L0]], {{.*}}%[[L1]]
; CHECK-DAG:   %[[P1:[0-9]+]]:hlxllr = LX_combine_w {{.*}}%[[L2]], {{.*}}%[[L3]]
; CHECK-DAG:   %[[R1:[0-9]+]]:intregs = A2_tfrsi 1
; CHECK-DAG:   %[[H0:[0-9]+]]:hvxvr = V6_lvsplatb {{.*}}%[[R1]]
; CHECK-DAG:   %[[R2:[0-9]+]]:intregs = A2_tfrsi 2
; CHECK-DAG:   %[[H1:[0-9]+]]:hvxvr = V6_lvsplatb {{.*}}%[[R2]]
; CHECK-DAG:   $v0 = COPY %[[H0]]
; CHECK-DAG:   $v1 = COPY %[[H1]]
; CHECK-DAG:   $ll0 = COPY %[[P0]]
; CHECK-DAG:   $ll1 = COPY %[[P1]]
; CHECK:       J2_call @para_pair_hvx_hlx_pair, hexagoncsr_hlx,
; CHECK-SAME:  implicit $v0, implicit $v1, implicit $ll0, implicit $ll1
; para_interleave_hvx_hlx_pair(v0=3, ll0=combine(4,5), v1=4, ll1=combine(6,7))
; CHECK-DAG:   %[[L4:[0-9]+]]:hlxlr = LX_assign_u 4
; CHECK-DAG:   %[[L5:[0-9]+]]:hlxlr = LX_assign_u 5
; CHECK-DAG:   %[[L6:[0-9]+]]:hlxlr = LX_assign_u 6
; CHECK-DAG:   %[[L7:[0-9]+]]:hlxlr = LX_assign_u 7
; CHECK-DAG:   %[[P2:[0-9]+]]:hlxllr = LX_combine_w {{.*}}%[[L4]], {{.*}}%[[L5]]
; CHECK-DAG:   %[[P3:[0-9]+]]:hlxllr = LX_combine_w {{.*}}%[[L6]], {{.*}}%[[L7]]
; CHECK-DAG:   %[[R3:[0-9]+]]:intregs = A2_tfrsi 3
; CHECK-DAG:   %[[H2:[0-9]+]]:hvxvr = V6_lvsplatb {{.*}}%[[R3]]
; CHECK-DAG:   %[[R4:[0-9]+]]:intregs = A2_tfrsi 4
; CHECK-DAG:   %[[H3:[0-9]+]]:hvxvr = V6_lvsplatb {{.*}}%[[R4]]
; CHECK-DAG:   $v0 = COPY %[[H2]]
; CHECK-DAG:   $ll0 = COPY %[[P2]]
; CHECK-DAG:   $v1 = COPY %[[H3]]
; CHECK-DAG:   $ll1 = COPY %[[P3]]
; CHECK:       J2_call @para_interleave_hvx_hlx_pair, hexagoncsr_hlx,
; CHECK-SAME:  implicit $v0, implicit $ll0, implicit $v1, implicit $ll1
define dso_local i32 @main() local_unnamed_addr #1 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatb.128B(i32 1)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatb.128B(i32 2)
  %2 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 0)
  %3 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 1)
  %4 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 2)
  %5 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 3)
  %6 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %2, <1024 x i16> %3)
  %7 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %4, <1024 x i16> %5)
  %call = tail call inreg <32 x i32> @para_pair_hvx_hlx_pair(<32 x i32> %0, <32 x i32> %1, <1024 x i32> %6, <1024 x i32> %7)
  %8 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 4)
  %9 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 5)
  %10 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 6)
  %11 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 7)
  %12 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatb.128B(i32 3)
  %13 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatb.128B(i32 4)
  %14 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %8, <1024 x i16> %9)
  %15 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %10, <1024 x i16> %11)
  %call2 = tail call inreg <32 x i32> @para_interleave_hvx_hlx_pair(<32 x i32> %12, <1024 x i32> %14, <32 x i32> %13, <1024 x i32> %15)
  ret i32 0
}

declare <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32>, <1024 x i32>) #2
declare <1024 x i16> @llvm.hexagon.LX.hi(<1024 x i32>) #2
declare <1024 x i16> @llvm.hexagon.LX.lo(<1024 x i32>) #2
declare <32 x i32> @llvm.hexagon.V6.vaddb.128B(<32 x i32>, <32 x i32>) #2
declare <1024 x i16> @llvm.hexagon.LX.assign.u(i32 immarg) #2
declare <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16>, <1024 x i16>) #2
declare <32 x i32> @llvm.hexagon.V6.lvsplatb.128B(i32) #2

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { nofree nounwind }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }


!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !4, i64 0}
!9 = !{!4, !4, i64 0}
