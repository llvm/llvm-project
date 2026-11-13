; Check that scalar args land in $r registers and HLX args land in $l registers,
; independently of each other, for two different argument orderings:
;   scalar_interleaved: (i32, hlx, i32, hlx) — interleaved scalar/HLX
;   scalar_then_hlx:    (i32, i32, hlx, hlx) — all scalars before all HLX

; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx -start-before=hexagon-isel -stop-after=hexagon-isel -verify-machineinstrs -o - %s | FileCheck %s

@.str = private unnamed_addr constant [13 x i8] c"s0=%d s1=%d\0A\00", align 1

; CHECK-LABEL: name: scalar_interleaved
; CHECK:       liveins: $r0, $l0, $r1, $l1
define dso_local inreg <1024 x i16> @scalar_interleaved(i32 %s0, <1024 x i16> %v0, i32 %s1, <1024 x i16> %v1) local_unnamed_addr #0 {
entry:
  %call = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @.str, i32 %s0, i32 %s1) #2
  %0 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %v0, <1024 x i16> %v1)
  ret <1024 x i16> %0
}

; CHECK-LABEL: name: scalar_then_hlx
; CHECK:       liveins: $r0, $r1, $l0, $l1
define dso_local inreg <1024 x i16> @scalar_then_hlx(i32 %s0, i32 %s1, <1024 x i16> %v0, <1024 x i16> %v1) local_unnamed_addr #0 {
entry:
  %call = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @.str, i32 %s0, i32 %s1) #2
  %0 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %v0, <1024 x i16> %v1)
  ret <1024 x i16> %0
}

; CHECK-LABEL: name: main
; --- scalar_interleaved call: (R0=10, L0=0, R1=20, L1=1) ---
; CHECK-DAG:   %[[L0:[0-9]+]]:hlxlr = LX_assign_u 0
; CHECK-DAG:   %[[L1:[0-9]+]]:hlxlr = LX_assign_u 1
; CHECK-DAG:   %[[S0:[0-9]+]]:intregs = A2_tfrsi 10
; CHECK-DAG:   %[[S1:[0-9]+]]:intregs = A2_tfrsi 20
; CHECK:       $r0 = COPY %[[S0]]
; CHECK:       $l0 = COPY %[[L0]]
; CHECK:       $r1 = COPY %[[S1]]
; CHECK:       $l1 = COPY %[[L1]]
; CHECK:       J2_call @scalar_interleaved, hexagoncsr_hlx,
; CHECK-SAME:  implicit $r0, implicit $l0, implicit $r1, implicit $l1
; --- scalar_then_hlx call: (R0=30, R1=40, L0=2, L1=3) ---
; CHECK-DAG:   %[[L2:[0-9]+]]:hlxlr = LX_assign_u 2
; CHECK-DAG:   %[[L3:[0-9]+]]:hlxlr = LX_assign_u 3
; CHECK-DAG:   %[[S2:[0-9]+]]:intregs = A2_tfrsi 30
; CHECK-DAG:   %[[S3:[0-9]+]]:intregs = A2_tfrsi 40
; CHECK:       $r0 = COPY %[[S2]]
; CHECK:       $r1 = COPY %[[S3]]
; CHECK:       $l0 = COPY %[[L2]]
; CHECK:       $l1 = COPY %[[L3]]
; CHECK:       J2_call @scalar_then_hlx, hexagoncsr_hlx,
; CHECK-SAME:  implicit $r0, implicit $r1, implicit $l0, implicit $l1
define dso_local i32 @main() local_unnamed_addr #1 {
entry:
  %0 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 0)
  %1 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 1)
  %call = tail call inreg <1024 x i16> @scalar_interleaved(i32 10, <1024 x i16> %0, i32 20, <1024 x i16> %1)
  %2 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 2)
  %3 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 3)
  %call2 = tail call inreg <1024 x i16> @scalar_then_hlx(i32 30, i32 40, <1024 x i16> %2, <1024 x i16> %3)
  ret i32 0
}

declare dso_local i32 @printf(ptr readonly captures(none), ...) local_unnamed_addr #1
declare <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16>, <1024 x i16>) #3
declare <1024 x i16> @llvm.hexagon.LX.assign.u(i32 immarg) #3

attributes #0 = { nofree noinline nounwind }
attributes #1 = { nofree nounwind }
attributes #2 = { nounwind }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
