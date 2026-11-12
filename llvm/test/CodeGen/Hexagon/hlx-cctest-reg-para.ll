; Passing 12 parameters to HLX function
; Verify that 12 parameters are passed over registers (all $l0-$l11 are live).

; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx -start-before=hexagon-isel -stop-after=hexagon-isel -verify-machineinstrs -o - %s | FileCheck %s --check-prefix=CHECK-ISEL

; CHECK-ISEL-LABEL: name: parameter_12
; CHECK-ISEL: liveins: $l0, $l1, $l2, $l3, $l4, $l5, $l6, $l7, $l8, $l9, $l10, $l11

; CHECK-ISEL-LABEL: name: main
; CHECK-ISEL: J2_call @parameter_12, hexagoncsr_hlx,
; CHECK-ISEL-SAME: implicit $l0, implicit $l1, implicit $l2, implicit $l3, implicit $l4, implicit $l5, implicit $l6, implicit $l7, implicit $l8, implicit $l9, implicit $l10, implicit $l11

target triple = "hexagon"

define dso_local inreg <1024 x i16> @parameter_12(<1024 x i16> %v0, <1024 x i16> %v1, <1024 x i16> %v2, <1024 x i16> %v3, <1024 x i16> %v4, <1024 x i16> %v5, <1024 x i16> %v6, <1024 x i16> %v7, <1024 x i16> %v8, <1024 x i16> %v9, <1024 x i16> %v10, <1024 x i16> %v11) local_unnamed_addr #0 {
entry:
    %0 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %v0, <1024 x i16> %v1)
    %1 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %v2, <1024 x i16> %v3)
    %2 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %v4, <1024 x i16> %v5)
    %3 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %v6, <1024 x i16> %v7)
    %4 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %v8, <1024 x i16> %v9)
    %5 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %v10, <1024 x i16> %v11)
    %6 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %0, <1024 x i16> %1)
    %7 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %2, <1024 x i16> %3)
    %8 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %4, <1024 x i16> %5)
    %9 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %6, <1024 x i16> %7)
    %10 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %8, <1024 x i16> %9)
    ret <1024 x i16> %10
}

define dso_local i32 @main() local_unnamed_addr #2 {
entry:
  %1 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 0)
  %2 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 1)
  %3 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 2)
  %4 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 3)
  %5 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 4)
  %call = tail call inreg <1024 x i16> @parameter_12(<1024 x i16> %1, <1024 x i16> %2, <1024 x i16> %3, <1024 x i16> %4, <1024 x i16> %5, <1024 x i16> %1, <1024 x i16> %2, <1024 x i16> %3, <1024 x i16> %4, <1024 x i16> %5, <1024 x i16> %1, <1024 x i16> %2)
  ret i32 0
}

declare <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16>, <1024 x i16>) #1
declare <1024 x i16> @llvm.hexagon.LX.assign.u(i32 immarg) #1

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nofree nounwind }


