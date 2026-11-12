; Passing 6 HLX_VectorPair parameters to HLX function
; Verify that 6 vector pair parameters are passed over registers ($ll0-$ll5 are live).

; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx -start-before=hexagon-isel -stop-after=hexagon-isel -verify-machineinstrs -o - %s | FileCheck %s --check-prefix=CHECK-ISEL

; CHECK-ISEL-LABEL: name: parameter_12_pair
; CHECK-ISEL: liveins: $ll0, $ll1, $ll2, $ll3, $ll4, $ll5

; CHECK-ISEL-LABEL: name: main
; CHECK-ISEL: J2_call @parameter_12_pair, hexagoncsr_hlx,
; CHECK-ISEL-SAME: implicit $ll0, implicit $ll1, implicit $ll2, implicit $ll3, implicit $ll4, implicit $ll5

target triple = "hexagon"


define dso_local inreg <1024 x i32> @parameter_12_pair(<1024 x i32> %p0, <1024 x i32> %p1, <1024 x i32> %p2, <1024 x i32> %p3, <1024 x i32> %p4, <1024 x i32> %p5) local_unnamed_addr #0 {
entry:
    %0 = tail call <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32> %p0, <1024 x i32> %p1)
    %1 = tail call <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32> %p2, <1024 x i32> %p3)
    %2 = tail call <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32> %p4, <1024 x i32> %p5)
    %3 = tail call <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32> %0, <1024 x i32> %1)
    %4 = tail call <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32> %3, <1024 x i32> %2)
    ret <1024 x i32> %4
}

define dso_local i32 @main() local_unnamed_addr #2 {
entry:
  %1 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 1)
  %2 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 0)
  %3 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 3)
  %4 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 2)
  %5 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 5)
  %6 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 4)
  %7 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 7)
  %8 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 6)
  %9 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 9)
  %10 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 8)
  %11 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 11)
  %12 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 10)
  %13 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %1, <1024 x i16> %2)
  %14 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %3, <1024 x i16> %4)
  %15 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %5, <1024 x i16> %6)
  %16 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %7, <1024 x i16> %8)
  %17 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %9, <1024 x i16> %10)
  %18 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %11, <1024 x i16> %12)
  %call = tail call inreg <1024 x i32> @parameter_12_pair(<1024 x i32> %13, <1024 x i32> %14, <1024 x i32> %15, <1024 x i32> %16, <1024 x i32> %17, <1024 x i32> %18)
  ret i32 0
}

declare <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32>, <1024 x i32>) #1
declare <1024 x i16> @llvm.hexagon.LX.assign.u(i32 immarg) #1
declare <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16>, <1024 x i16>) #1

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nofree nounwind }


