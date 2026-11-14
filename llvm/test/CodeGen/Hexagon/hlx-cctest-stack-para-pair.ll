; Pass 8 HLX vector pairs: the first 6 go in LL0-LL5, p6 and p7 go on the
; VTCM stack as two single-vector halves each.

; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx -stop-after=hexagon-isel -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK-ISEL
; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx -stop-after=prolog-epilog -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK-PEI
; The VTCM argument area and the incoming VTCM arguments survive MIR serialization.
; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx -stop-after=finalize-isel < %s | llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx -x mir -start-after=finalize-isel -stop-after=prolog-epilog -verify-machineinstrs | FileCheck %s --check-prefix=CHECK-PEI

; CHECK-ISEL-LABEL: name: parameter_16_pair
; CHECK-ISEL: liveins: $ll0, $ll1, $ll2, $ll3, $ll4, $ll5
; CHECK-ISEL-DAG: hlxlr = LX_L16b_pseudo {{.*}} :: (load (s16384) from %fixed-stack.0)
; CHECK-ISEL-DAG: hlxlr = LX_L16b_pseudo {{.*}} :: (load (s16384) from %fixed-stack.0 + 2048)
; CHECK-ISEL-DAG: hlxlr = LX_L16b_pseudo {{.*}} :: (load (s16384) from %fixed-stack.1)
; CHECK-ISEL-DAG: hlxlr = LX_L16b_pseudo {{.*}} :: (load (s16384) from %fixed-stack.1 + 2048)

; CHECK-ISEL-LABEL: name: main
; CHECK-ISEL-DAG: LX_S16b_pseudo {{.*}} :: (store (s16384) into stack)
; CHECK-ISEL-DAG: LX_S16b_pseudo {{.*}} :: (store (s16384) into stack + 2048)
; CHECK-ISEL-DAG: LX_S16b_pseudo {{.*}} :: (store (s16384) into stack + 4096)
; CHECK-ISEL-DAG: LX_S16b_pseudo {{.*}} :: (store (s16384) into stack + 6144)
; CHECK-ISEL: J2_call @parameter_16_pair, hexagoncsr_hlx
; CHECK-ISEL-SAME: implicit $ll0, implicit $ll1, implicit $ll2, implicit $ll3, implicit $ll4, implicit $ll5

; CHECK-PEI-LABEL: name: parameter_16_pair
; CHECK-PEI-NOT: V2_vallocframe_imm
; CHECK-PEI-DAG: A2_addi killed $r{{[0-9]+}}, 4096
; CHECK-PEI-DAG: A2_addi killed renamable $r{{[0-9]+}}, 2048
; CHECK-PEI-DAG: LX_L16b_pseudo {{.*}} :: (load (s16384) from %fixed-stack.0)
; CHECK-PEI-DAG: LX_L16b_pseudo {{.*}} :: (load (s16384) from %fixed-stack.0 + 2048)
; CHECK-PEI-DAG: LX_L16b_pseudo {{.*}} :: (load (s16384) from %fixed-stack.1)
; CHECK-PEI-DAG: LX_L16b_pseudo {{.*}} :: (load (s16384) from %fixed-stack.1 + 2048)
; CHECK-PEI: PS_jmpret

; CHECK-PEI-LABEL: name: main
; CHECK-PEI: frame-setup V2_vallocframe_imm 4
; CHECK-PEI: $[[VSP:r[0-9]+]] = COPY $vsp
; CHECK-PEI-DAG: A2_addi renamable $[[VSP]], 6144
; CHECK-PEI-DAG: A2_addi renamable $[[VSP]], 4096
; CHECK-PEI-DAG: A2_addi renamable $[[VSP]], 2048
; CHECK-PEI-DAG: LX_S16b_pseudo killed renamable $[[VSP]], {{.*}} :: (store (s16384) into stack)
; CHECK-PEI-DAG: LX_S16b_pseudo {{.*}} :: (store (s16384) into stack + 2048)
; CHECK-PEI-DAG: LX_S16b_pseudo {{.*}} :: (store (s16384) into stack + 4096)
; CHECK-PEI-DAG: LX_S16b_pseudo {{.*}} :: (store (s16384) into stack + 6144)
; CHECK-PEI: J2_call @parameter_16_pair, hexagoncsr_hlx
; CHECK-PEI: V2_vdeallocframe_imm 4

define <1024 x i32> @parameter_16_pair(<1024 x i32> %p0, <1024 x i32> %p1, <1024 x i32> %p2, <1024 x i32> %p3, <1024 x i32> %p4, <1024 x i32> %p5, <1024 x i32> %p6, <1024 x i32> %p7) {
entry:
    %0 = tail call <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32> %p0, <1024 x i32> %p1)
    %1 = tail call <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32> %p2, <1024 x i32> %p3)
    %2 = tail call <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32> %p4, <1024 x i32> %p5)
    %3 = tail call <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32> %p6, <1024 x i32> %p7)
    %4 = tail call <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32> %0, <1024 x i32> %1)
    %5 = tail call <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32> %2, <1024 x i32> %3)
    %6 = tail call <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32> %4, <1024 x i32> %5)
    ret <1024 x i32> %6
}

define i32 @main() {
entry:
  %0 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 0)
  %1 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 1)
  %2 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 2)
  %3 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 3)
  %4 = tail call <1024 x i16> @llvm.hexagon.LX.assign.u(i32 4)
  %p0 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %1, <1024 x i16> %0)
  %p1 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %3, <1024 x i16> %2)
  %p2 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %0, <1024 x i16> %4)
  %p3 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %2, <1024 x i16> %1)
  %p4 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %4, <1024 x i16> %3)
  %p5 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %1, <1024 x i16> %2)
  %p6 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %3, <1024 x i16> %4)
  %p7 = tail call <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16> %0, <1024 x i16> %1)
  %call = tail call <1024 x i32> @parameter_16_pair(<1024 x i32> %p0, <1024 x i32> %p1, <1024 x i32> %p2, <1024 x i32> %p3, <1024 x i32> %p4, <1024 x i32> %p5, <1024 x i32> %p6, <1024 x i32> %p7)
  ret i32 0
}

declare <1024 x i16> @llvm.hexagon.LX.assign.u(i32 immarg)
declare <1024 x i32> @llvm.hexagon.LX.combine.w(<1024 x i16>, <1024 x i16>)
declare <1024 x i32> @llvm.hexagon.LX.add.w(<1024 x i32>, <1024 x i32>)
