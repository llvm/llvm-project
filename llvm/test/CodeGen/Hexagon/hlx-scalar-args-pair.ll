; Check that scalar args land in $r registers and HLX_VectorPair args land in $ll registers,
; independently of each other, for two different argument orderings:
;   scalar_interleaved_pair: (i32, hlx_pair, i32, hlx_pair) — interleaved scalar/HLX_VectorPair
;   scalar_then_hlx_pair:    (i32, i32, hlx_pair, hlx_pair) — all scalars before all HLX_VectorPair

; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx -start-before=hexagon-isel -stop-after=hexagon-isel -verify-machineinstrs -o - %s | FileCheck %s

@.str = private unnamed_addr constant [13 x i8] c"s0=%d s1=%d\0A\00", align 1

; CHECK-LABEL: name: scalar_interleaved_pair
; CHECK:       liveins: $r0, $ll0, $r1, $ll1
define dso_local inreg <1024 x i32> @scalar_interleaved_pair(i32 %s0, <1024 x i32> %p0, i32 %s1, <1024 x i32> %p1) local_unnamed_addr #0 {
entry:
  %call = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @.str, i32 %s0, i32 %s1) #2
  %0 = tail call <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32> %p0, <1024 x i32> %p1)
  ret <1024 x i32> %0
}

; CHECK-LABEL: name: scalar_then_hlx_pair
; CHECK:       liveins: $r0, $r1, $ll0, $ll1
define dso_local inreg <1024 x i32> @scalar_then_hlx_pair(i32 %s0, i32 %s1, <1024 x i32> %p0, <1024 x i32> %p1) local_unnamed_addr #0 {
entry:
  %call = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @.str, i32 %s0, i32 %s1) #2
  %0 = tail call <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32> %p0, <1024 x i32> %p1)
  ret <1024 x i32> %0
}

; CHECK-LABEL: name: main
; --- scalar_interleaved_pair call: (R0=10, LL0=(0,1), R1=20, LL1=(2,3)) ---
; CHECK-DAG:   %[[L0:[0-9]+]]:hlxlr = LX_assign_u 0
; CHECK-DAG:   %[[L1:[0-9]+]]:hlxlr = LX_assign_u 1
; CHECK-DAG:   %[[L2:[0-9]+]]:hlxlr = LX_assign_u 2
; CHECK-DAG:   %[[L3:[0-9]+]]:hlxlr = LX_assign_u 3
; CHECK-DAG:   %[[LL0:[0-9]+]]:hlxllr = LX_combine_w {{.*}}%[[L0]], {{.*}}%[[L1]]
; CHECK-DAG:   %[[LL1:[0-9]+]]:hlxllr = LX_combine_w {{.*}}%[[L2]], {{.*}}%[[L3]]
; CHECK-DAG:   %[[S0:[0-9]+]]:intregs = A2_tfrsi 10
; CHECK-DAG:   %[[S1:[0-9]+]]:intregs = A2_tfrsi 20
; CHECK-DAG:       $r0 = COPY %[[S0]]
; CHECK-DAG:       $ll0 = COPY %[[LL0]]
; CHECK-DAG:       $r1 = COPY %[[S1]]
; CHECK-DAG:       $ll1 = COPY %[[LL1]]
; CHECK:       J2_call @scalar_interleaved_pair, hexagoncsr_hlx,
; CHECK-SAME:  implicit $r0, implicit $ll0, implicit $r1, implicit $ll1
; --- scalar_then_hlx_pair call: (R0=30, R1=40, LL0=(4,5), LL1=(6,7)) ---
; CHECK-DAG:   %[[L4:[0-9]+]]:hlxlr = LX_assign_u 4
; CHECK-DAG:   %[[L5:[0-9]+]]:hlxlr = LX_assign_u 5
; CHECK-DAG:   %[[L6:[0-9]+]]:hlxlr = LX_assign_u 6
; CHECK-DAG:   %[[L7:[0-9]+]]:hlxlr = LX_assign_u 7
; CHECK-DAG:   %[[LL2:[0-9]+]]:hlxllr = LX_combine_w {{.*}}%[[L4]], {{.*}}%[[L5]]
; CHECK-DAG:   %[[LL3:[0-9]+]]:hlxllr = LX_combine_w {{.*}}%[[L6]], {{.*}}%[[L7]]
; CHECK-DAG:   %[[S2:[0-9]+]]:intregs = A2_tfrsi 30
; CHECK-DAG:   %[[S3:[0-9]+]]:intregs = A2_tfrsi 40
; CHECK-DAG:       $r0 = COPY %[[S2]]
; CHECK-DAG:       $r1 = COPY %[[S3]]
; CHECK-DAG:       $ll0 = COPY %[[LL2]]
; CHECK-DAG:       $ll1 = COPY %[[LL3]]
; CHECK:       J2_call @scalar_then_hlx_pair, hexagoncsr_hlx,
; CHECK-SAME:  implicit $r0, implicit $r1, implicit $ll0, implicit $ll1
define dso_local i32 @main() local_unnamed_addr #1 {
entry:
  %0 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 0)
  %1 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 1)
  %2 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 2)
  %3 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 3)
  %4 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %0, <1024 x i16> %1)
  %5 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %2, <1024 x i16> %3)
  %call = tail call inreg <1024 x i32> @scalar_interleaved_pair(i32 10, <1024 x i32> %4, i32 20, <1024 x i32> %5)
  %6 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 4)
  %7 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 5)
  %8 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 6)
  %9 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 7)
  %10 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %6, <1024 x i16> %7)
  %11 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %8, <1024 x i16> %9)
  %call2 = tail call inreg <1024 x i32> @scalar_then_hlx_pair(i32 30, i32 40, <1024 x i32> %10, <1024 x i32> %11)
  ret i32 0
}

declare dso_local i32 @printf(ptr readonly captures(none), ...) local_unnamed_addr #1
declare <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32>, <1024 x i32>) #3
declare <1024 x i16> @llvm.hexagon.LX.assign.u(i32 immarg) #3
declare <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16>, <1024 x i16>) #3

attributes #0 = { nofree noinline nounwind }
attributes #1 = { nofree nounwind }
attributes #2 = { nounwind }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }

