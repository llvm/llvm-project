; Check that HLX callee saved register spills are generated correctly

; RUN: llc -march=hexagon -mattr=+hlx -mcpu=hexagonv85 -stop-after="prolog-epilog" < %s -o - | FileCheck %s

; Only the callee-saved L12-L15 are saved, each in its own VTCM stack slot.
; CHECK-LABEL: name: hlx_use_all_L16_regs
; CHECK: V2_vallocframe_imm 4
; CHECK-NOT: PS_storerl
; CHECK: PS_storerl killed $[[R0:r[0-9]+]], 6144, killed $l12 :: (store (s16384) into %stack.[[S12:[0-9]+]])
; CHECK: PS_storerl killed $[[R1:r[0-9]+]], 4096, killed $l13 :: (store (s16384) into %stack.[[S13:[0-9]+]])
; CHECK: PS_storerl killed $[[R2:r[0-9]+]], 2048, killed $l14 :: (store (s16384) into %stack.[[S14:[0-9]+]])
; CHECK: PS_storerl killed $[[R3:r[0-9]+]], 0, killed $l15 :: (store (s16384) into %stack.[[S15:[0-9]+]])
; CHECK-NOT: PS_storerl
; CHECK: $l12 = PS_loadrl killed $[[R4:r[0-9]+]], 6144 :: (load (s16384) from %stack.[[S12]])
; CHECK: $l13 = PS_loadrl killed $[[R5:r[0-9]+]], 4096 :: (load (s16384) from %stack.[[S13]])
; CHECK: $l14 = PS_loadrl killed $[[R6:r[0-9]+]], 2048 :: (load (s16384) from %stack.[[S14]])
; CHECK: $l15 = PS_loadrl killed $[[R7:r[0-9]+]], 0 :: (load (s16384) from %stack.[[S15]])
; CHECK-NOT: PS_loadrl
; CHECK: V2_vdeallocframe_imm 4
define <1024 x i16> @hlx_use_all_L16_regs() nounwind #0 {
  %1 = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 1)
  %2 = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 2)
  %3 = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 3)
  %4 = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 4)
  %5 = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 5)
  %6 = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 6)
  %7 = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 7)
  %8 = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 8)
  %9 = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 9)
  %10 = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 10)
  %11 = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 11)
  %12 = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 12)
  %13 = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 13)
  %14 = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 14)
  %15 = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 15)
  %16 = tail call <1024 x i16> @llvm.hexagon.LX.assign.r(i32 16)
  %17 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %1, <1024 x i16> %2)
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
  %32 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %16, <1024 x i16> %31)
  ret <1024 x i16> %32
}

; Only L12, which holds a value across the call, is saved.
; CHECK-LABEL: name: bar_L
; CHECK: V2_vallocframe_imm 1
; CHECK: $[[REG0:r[0-9]+]] = A2_tfrcrr $vsp
; CHECK: PS_storerl killed $[[REG0]], 0, killed $l12
; CHECK-NOT: PS_storerl
; CHECK: $l12 = LX_add_h
; CHECK: J2_call @hlx_use_all_L16_regs, hexagoncsr_hlx,
; CHECK: $[[REG1:r[0-9]+]] = A2_tfrcrr $vsp
; CHECK: $l12 = PS_loadrl killed $[[REG1]], 0
; CHECK-NOT: PS_loadrl
; CHECK: V2_vdeallocframe_imm 1
define <1024 x i16> @bar_L(ptr nocapture noundef readonly %src0, ptr nocapture noundef readonly %src1) {
entry:
  %0 = load <1024 x i16>, ptr %src0, align 2048
  %1 = load <1024 x i16>, ptr %src1, align 2048
  %2 = tail call <1024 x i16> @llvm.hexagon.LX.add.h(<1024 x i16> %0, <1024 x i16> %1)
  call <1024 x i16> @hlx_use_all_L16_regs()
  ret <1024 x i16> %2
}

attributes #0 = { optnone noinline }
