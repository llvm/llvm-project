; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64-unknown-unknown < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s \
; RUN:  --check-prefix=CHECK-P7
; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mtriple=powerpc64-unknown-unknown < %s | FileCheck %s

define signext i32 @f32toi32(float %a) {
entry:
  %0 = bitcast float %a to i32
  ret i32 %0
; CHECK-P7: stfs 1,
; CHECK-P7: lwa 3,
; CHECK: xscvdpspn [[CONVREG:[0-9]+]], 1
; CHECK: xxsldwi [[SHIFTREG:[0-9]+]], [[CONVREG]], [[CONVREG]], 3
; CHECK: mfvsrwz 3, [[SHIFTREG]]
}

define i64 @f64toi64(double %a) {
entry:
  %0 = bitcast double %a to i64
  ret i64 %0
; CHECK-P7: stxsdx 1,
; CHECK-P7: ld 3,
; CHECK: mffprd 3, 1
}

define float @i32tof32(i32 signext %a) {
entry:
  %0 = bitcast i32 %a to float
  ret float %0
; CHECK-P7: stw 3,
; CHECK-P7: lfs 1,
; CHECK: mtvsrd [[MOVEREG:[0-9]+]], 3
; CHECK: xxsldwi [[SHIFTREG:[0-9]+]], [[MOVEREG]], [[MOVEREG]], 1
; CHECK: xscvspdpn 1, [[SHIFTREG]]
}

define double @i64tof64(i64 %a) {
entry:
  %0 = bitcast i64 %a to double
  ret double %0
; CHECK-P7: std 3,
; CHECK-P7: lxsdx 1,
; CHECK: mtvsrd 1, 3
}

define zeroext i32 @f32toi32u(float %a) {
entry:
  %0 = bitcast float %a to i32
  ret i32 %0
; CHECK-P7: stfs 1,
; CHECK-P7: lwz 3,
; CHECK: xscvdpspn [[CONVREG:[0-9]+]], 1
; CHECK: xxsldwi [[SHIFTREG:[0-9]+]], [[CONVREG]], [[CONVREG]], 3
; CHECK: mfvsrwz 3, [[SHIFTREG]]
}

define i64 @f64toi64u(double %a) {
entry:
  %0 = bitcast double %a to i64
  ret i64 %0
; CHECK-P7: stxsdx 1,
; CHECK-P7: ld 3,
; CHECK: mffprd 3, 1
}

define float @i32utof32(i32 zeroext %a) {
entry:
  %0 = bitcast i32 %a to float
  ret float %0
; CHECK-P7: stw 3,
; CHECK-P7: lfs 1,
; CHECK: mtvsrd [[MOVEREG:[0-9]+]], 3
; CHECK: xxsldwi [[SHIFTREG:[0-9]+]], [[MOVEREG]], [[MOVEREG]], 1
; CHECK: xscvspdpn 1, [[SHIFTREG]]
}

define double @i64utof64(i64 %a) {
entry:
  %0 = bitcast i64 %a to double
  ret double %0
; CHECK-P7: std 3,
; CHECK-P7: lxsdx 1,
; CHECK: mtvsrd 1, 3
}
