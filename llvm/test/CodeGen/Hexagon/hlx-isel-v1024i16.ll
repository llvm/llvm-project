; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx -stop-after="hexagon-isel" -verify-machineinstrs < %s | FileCheck %s

; Check that vector operations of type v1024i16 are lowered
; to HLX instructions

target triple = "hexagon"

; CHECK-LABEL: name: hlx_add_sub
; CHECK: LX_L16b_pseudo
; CHECK-NEXT: LX_L16b_pseudo
; CHECK-NEXT: LX_add_h
; CHECK-NEXT: LX_assign_s
; CHECK-NEXT: LX_sub_h
; CHECK-NEXT: LX_add_h_sat
; CHECK-NEXT: LX_sub_h_sat
; CHECK-NEXT: LX_add_uh_sat
; CHECK-NEXT: LX_sub_uh_sat
; CHECK-NEXT: LX_S16b_pseudo

define dso_local void @hlx_add_sub(ptr %din0, ptr %din1, ptr %dout) {
entry:
  %0 = load <1024 x i16>, ptr %din0, align 2048
  %1 = load <1024 x i16>, ptr %din1, align 2048
  %2 = add <1024 x i16> %1, %0
  %3 = sub <1024 x i16> %2, splat (i16 10)
  %4 = call <1024 x i16> @llvm.sadd.sat.v1024i16(<1024 x i16> %2, <1024 x i16> %3)
  %5 = call <1024 x i16> @llvm.ssub.sat.v1024i16(<1024 x i16> %3, <1024 x i16> %4)
  %6 = call <1024 x i16> @llvm.uadd.sat.v1024i16(<1024 x i16> %5, <1024 x i16> %4)
  %7 = call <1024 x i16> @llvm.usub.sat.v1024i16(<1024 x i16> %6, <1024 x i16> %5)
  store <1024 x i16> %7, ptr %dout, align 2048
  ret void
}


; CHECK-LABEL: name: hlx_bitwise
; CHECK: LX_and
; CHECK-NEXT: LX_or
; CHECK-NEXT: LX_xor
; CHECK-NEXT: LX_shr_uh_uh
define dso_local void @hlx_bitwise(ptr %din0, ptr %din1, ptr %dout) {
entry:
  %0 = load <1024 x i16>, ptr %din0, align 2048
  %1 = load <1024 x i16>, ptr %din1, align 2048
  %2 = and <1024 x i16> %1, %0
  %3 = or <1024 x i16> %1, %0
  %4 = xor <1024 x i16> %3, %2
  %5 = lshr <1024 x i16> %3, %4
  store <1024 x i16> %5, ptr %dout, align 2048
  ret void
}

; CHECK-LABEL: name: hlx_min_max
; CHECK: LX_max_h
; CHECK-NEXT: LX_min_h
; CHECK-NEXT: LX_min_uh
; CHECK-NEXT: LX_max_uh
define dso_local void @hlx_min_max(ptr %din0, ptr %din1, ptr %dout) #0 {
entry:
  %0 = load <1024 x i16>, ptr %din0, align 2048
  %1 = load <1024 x i16>, ptr %din1, align 2048
  %2 = call <1024 x i16> @llvm.smax.v1024i16(<1024 x i16> %1, <1024 x i16> %0)
  %3 = call <1024 x i16> @llvm.smin.v1024i16(<1024 x i16> %2, <1024 x i16> %0)
  %4 = call <1024 x i16> @llvm.umin.v1024i16(<1024 x i16> %3, <1024 x i16> %1)
  %5 = call <1024 x i16> @llvm.umax.v1024i16(<1024 x i16> %4, <1024 x i16> %3)
  store <1024 x i16> %5, ptr %dout, align 2048
  ret void
}

; CHECK-LABEL: name: hlx_abs
; CHECK: LX_abs_uh
define dso_local void @hlx_abs(ptr %din0, ptr %dout) {
entry:
  %0 = load <1024 x i16>, ptr %din0, align 2048
  %1 = call <1024 x i16> @llvm.abs.v1024i16(<1024 x i16> %0, i1 true)
  store <1024 x i16> %1, ptr %dout, align 2048
  ret void
}

declare <1024 x i16> @llvm.sadd.sat.v1024i16(<1024 x i16> %a, <1024 x i16> %b)
declare <1024 x i16> @llvm.ssub.sat.v1024i16(<1024 x i16> %a, <1024 x i16> %b)
declare <1024 x i16> @llvm.uadd.sat.v1024i16(<1024 x i16> %a, <1024 x i16> %b)
declare <1024 x i16> @llvm.usub.sat.v1024i16(<1024 x i16> %a, <1024 x i16> %b)
declare <1024 x i16> @llvm.smax.v1024i16(<1024 x i16> %a, <1024 x i16> %b)
declare <1024 x i16> @llvm.smin.v1024i16(<1024 x i16> %a, <1024 x i16> %b)
declare <1024 x i16> @llvm.umax.v1024i16(<1024 x i16> %a, <1024 x i16> %b)
declare <1024 x i16> @llvm.umin.v1024i16(<1024 x i16> %a, <1024 x i16> %b)
declare <1024 x i16> @llvm.abs.v1024i16(<1024 x i16> %a, i1 %b)

attributes #0 = { optnone noinline }
