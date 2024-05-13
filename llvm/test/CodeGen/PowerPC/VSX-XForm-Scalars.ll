; RUN: llc < %s -mcpu=pwr8 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names -verify-machineinstrs \
; RUN:   | FileCheck %s --check-prefix=CHECK-P8
; RUN: llc < %s -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names -verify-machineinstrs \
; RUN:   | FileCheck %s --check-prefix=CHECK-P9

@a = external local_unnamed_addr global <4 x i32>, align 16
@pb = external local_unnamed_addr global ptr, align 8

define void @testExpandPostRAPseudo(ptr nocapture readonly %ptr) {
; CHECK-P8-LABEL: testExpandPostRAPseudo:
; CHECK-P8:  # %bb.0: # %entry
; CHECK-P8:    lfiwzx f0, 0, r3
; CHECK-P8:    ld r4, .LC0@toc@l(r4)
; CHECK-P8:    xxspltw vs0, vs0, 1
; CHECK-P8;    stxvd2x vs0, 0, r4
; CHECK-P8:    lfiwax f0, 0, r3
; CHECK-P8:    addis r3, r2, .LC1@toc@ha
; CHECK-P8:    ld r3, .LC1@toc@l(r3)
; CHECK-P8:    xscvsxdsp f0, f0
; CHECK-P8:    ld r3, 0(r3)
; CHECK-P8:    stfsx f0, r3, r4
; CHECK-P8:    blr
;
; CHECK-P9-LABEL: testExpandPostRAPseudo:
; CHECK-P9:  # %bb.0: # %entry
; CHECK-P9:    addis r4, r2, .LC0@toc@ha
; CHECK-P9:    lxvwsx vs0, 0, r3
; CHECK-P9:    ld r4, .LC0@toc@l(r4)
; CHECK-P9:    stxv vs0, 0(r4)
; CHECK-P9:    lis r4, 1024
; CHECK-P9:    lfiwax f0, 0, r3
; CHECK-P9:    addis r3, r2, .LC1@toc@ha
; CHECK-P9:    ld r3, .LC1@toc@l(r3)
; CHECK-P9:    xscvsxdsp f0, f0
; CHECK-P9:    ld r3, 0(r3)
; CHECK-P9:    stfsx f0, r3, r4
; CHECK-P9:    blr
entry:
  %0 = load i32, ptr %ptr, align 4
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %0, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  store <4 x i32> %splat.splat, ptr @a, align 16
  tail call void asm sideeffect "#Clobber Rigisters", "~{f0},~{f3},~{f4},~{f5},~{f6},~{f7},~{f8},~{f9},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15},~{f16},~{f17},~{f18},~{f19},~{f20},~{f21},~{f22},~{f23},~{f24},~{f25},~{f26},~{f27},~{f28},~{f29},~{f30},~{f31}"()
  %1 = load i32, ptr %ptr, align 4
  %conv = sitofp i32 %1 to float
  %2 = load ptr, ptr @pb, align 8
  %add.ptr = getelementptr inbounds float, ptr %2, i64 16777216
  store float %conv, ptr %add.ptr, align 4
  ret void
}
