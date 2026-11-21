; Check that a preserve_hlx function saves and restores every HLX register it
; clobbers, including the argument registers L0-L11.

; RUN: llc -mtriple=hexagon -mcpu=hexagonv85 -mattr=+hlx -stop-after=prolog-epilog \
; RUN:   < %s | FileCheck %s

; CHECK-LABEL: name: hlx_use_all_L16_regs
; CHECK: frame-setup V2_vallocframe_imm 16,
; CHECK-DAG: PS_storerl {{.*}}, killed $l0 ::
; CHECK-DAG: PS_storerl {{.*}}, killed $l1 ::
; CHECK-DAG: PS_storerl {{.*}}, killed $l2 ::
; CHECK-DAG: PS_storerl {{.*}}, killed $l3 ::
; CHECK-DAG: PS_storerl {{.*}}, killed $l4 ::
; CHECK-DAG: PS_storerl {{.*}}, killed $l5 ::
; CHECK-DAG: PS_storerl {{.*}}, killed $l6 ::
; CHECK-DAG: PS_storerl {{.*}}, killed $l7 ::
; CHECK-DAG: PS_storerl {{.*}}, killed $l8 ::
; CHECK-DAG: PS_storerl {{.*}}, killed $l9 ::
; CHECK-DAG: PS_storerl {{.*}}, killed $l10 ::
; CHECK-DAG: PS_storerl {{.*}}, killed $l11 ::
; CHECK-DAG: PS_storerl {{.*}}, killed $l12 ::
; CHECK-DAG: PS_storerl {{.*}}, killed $l13 ::
; CHECK-DAG: PS_storerl {{.*}}, killed $l14 ::
; CHECK-DAG: PS_storerl {{.*}}, killed $l15 ::
; CHECK: LX_S16b_pseudo
; CHECK-DAG: $l0 = PS_loadrl
; CHECK-DAG: $l1 = PS_loadrl
; CHECK-DAG: $l2 = PS_loadrl
; CHECK-DAG: $l3 = PS_loadrl
; CHECK-DAG: $l4 = PS_loadrl
; CHECK-DAG: $l5 = PS_loadrl
; CHECK-DAG: $l6 = PS_loadrl
; CHECK-DAG: $l7 = PS_loadrl
; CHECK-DAG: $l8 = PS_loadrl
; CHECK-DAG: $l9 = PS_loadrl
; CHECK-DAG: $l10 = PS_loadrl
; CHECK-DAG: $l11 = PS_loadrl
; CHECK-DAG: $l12 = PS_loadrl
; CHECK-DAG: $l13 = PS_loadrl
; CHECK-DAG: $l14 = PS_loadrl
; CHECK-DAG: $l15 = PS_loadrl
; CHECK: V2_vdeallocframe_imm 16,

define hexagon_preserve_hlx void @hlx_use_all_L16_regs(ptr writeonly %dst) #0 {
entry:
  %0 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 1)
  %1 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 2)
  %2 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 3)
  %3 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 4)
  %4 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 5)
  %5 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 6)
  %6 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 7)
  %7 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 8)
  %8 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 9)
  %9 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 10)
  %10 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 11)
  %11 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 12)
  %12 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 13)
  %13 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 14)
  %14 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 15)
  %15 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 16)
  %16 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %0, <1024 x i16> %1)
  %17 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %1, <1024 x i16> %16)
  %18 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %2, <1024 x i16> %17)
  %19 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %3, <1024 x i16> %18)
  %20 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %4, <1024 x i16> %19)
  %21 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %5, <1024 x i16> %20)
  %22 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %6, <1024 x i16> %21)
  %23 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %7, <1024 x i16> %22)
  %24 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %8, <1024 x i16> %23)
  %25 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %9, <1024 x i16> %24)
  %26 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %10, <1024 x i16> %25)
  %27 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %11, <1024 x i16> %26)
  %28 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %12, <1024 x i16> %27)
  %29 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %13, <1024 x i16> %28)
  %30 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %14, <1024 x i16> %29)
  %31 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %15, <1024 x i16> %30)
  store <1024 x i16> %31, ptr %dst, align 2048
  ret void
}

declare <1024 x i16> @llvm.hexagon.LX.assign.u(i32 immarg)

declare <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16>, <1024 x i16>)

attributes #0 = { optnone noinline }
