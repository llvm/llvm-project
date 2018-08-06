; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -enable-ppc-quad-precision -verify-machineinstrs \
; RUN:   -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s | FileCheck %s


define void @qp_trunc(fp128* nocapture readonly %a, fp128* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = tail call fp128 @llvm.trunc.f128(fp128 %0)
  store fp128 %1, fp128* %res, align 16
  ret void
; CHECK-LABEL: qp_trunc
; CHECK: xsrqpi 1, v{{[0-9]+}}, v{{[0-9]+}}, 1
; CHECK: blr
}
declare fp128     @llvm.trunc.f128(fp128 %Val)

define void @qp_rint(fp128* nocapture readonly %a, fp128* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = tail call fp128 @llvm.rint.f128(fp128 %0)
  store fp128 %1, fp128* %res, align 16
  ret void
; CHECK-LABEL: qp_rint
; CHECK: xsrqpix 0, v{{[0-9]+}}, v{{[0-9]+}}, 3
; CHECK: blr
}
declare fp128     @llvm.rint.f128(fp128 %Val)

define void @qp_nearbyint(fp128* nocapture readonly %a, fp128* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = tail call fp128 @llvm.nearbyint.f128(fp128 %0)
  store fp128 %1, fp128* %res, align 16
  ret void
; CHECK-LABEL: qp_nearbyint
; CHECK: xsrqpi 0, v{{[0-9]+}}, v{{[0-9]+}}, 3
; CHECK: blr
}
declare fp128     @llvm.nearbyint.f128(fp128 %Val)

define void @qp_round(fp128* nocapture readonly %a, fp128* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = tail call fp128 @llvm.round.f128(fp128 %0)
  store fp128 %1, fp128* %res, align 16
  ret void
; CHECK-LABEL: qp_round
; CHECK: xsrqpi 0, v{{[0-9]+}}, v{{[0-9]+}}, 0
; CHECK: blr
}
declare fp128     @llvm.round.f128(fp128 %Val)

define void @qp_floor(fp128* nocapture readonly %a, fp128* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = tail call fp128 @llvm.floor.f128(fp128 %0)
  store fp128 %1, fp128* %res, align 16
  ret void
; CHECK-LABEL: qp_floor
; CHECK: xsrqpi 1, v{{[0-9]+}}, v{{[0-9]+}}, 3
; CHECK: blr
}
declare fp128     @llvm.floor.f128(fp128 %Val)

define void @qp_ceil(fp128* nocapture readonly %a, fp128* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = tail call fp128 @llvm.ceil.f128(fp128 %0)
  store fp128 %1, fp128* %res, align 16
  ret void
; CHECK-LABEL: qp_ceil
; CHECK: xsrqpi 1, v{{[0-9]+}}, v{{[0-9]+}}, 2
; CHECK: blr
}
declare fp128     @llvm.ceil.f128(fp128 %Val)

