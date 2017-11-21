; RUN: llc < %s -mcpu=pwr8 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefix=CHECK-P8
; RUN: llc < %s -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefix=CHECK-P9

@a = external local_unnamed_addr global <4 x i32>, align 16
@pb = external local_unnamed_addr global float*, align 8

define void @testExpandPostRAPseudo(i32* nocapture readonly %ptr) {
; CHECK-P8-LABEL:     testExpandPostRAPseudo:
; CHECK-P8:           lxsiwax 34, 0, 3
; CHECK-P8-NEXT:      xxspltw 34, 34, 1
; CHECK-P8-NEXT:      stvx 2, 0, 4
; CHECK-P8:           #APP
; CHECK-P8-NEXT:      #Clobber Rigisters
; CHECK-P8-NEXT:      #NO_APP
; CHECK-P8-NEXT:      lis 4, 1024
; CHECK-P8-NEXT:      lfiwax 0, 0, 3
; CHECK-P8:           stfsx 0, 3, 4
; CHECK-P8-NEXT:      blr

; CHECK-P9-LABEL:     testExpandPostRAPseudo:
; CHECK-P9:           lxvwsx 0, 0, 3
; CHECK-P9:           stxvx 0, 0, 4
; CHECK-P9:           #APP
; CHECK-P9-NEXT:      #Clobber Rigisters
; CHECK-P9-NEXT:      #NO_APP
; CHECK-P9-NEXT:      lis 4, 1024
; CHECK-P9-NEXT:      lfiwax 0, 0, 3
; CHECK-P9:           stfsx 0, 3, 4
; CHECK-P9-NEXT:      blr

entry:
  %0 = load i32, i32* %ptr, align 4
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %0, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  store <4 x i32> %splat.splat, <4 x i32>* @a, align 16
  tail call void asm sideeffect "#Clobber Rigisters", "~{f0},~{f3},~{f4},~{f5},~{f6},~{f7},~{f8},~{f9},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15},~{f16},~{f17},~{f18},~{f19},~{f20},~{f21},~{f22},~{f23},~{f24},~{f25},~{f26},~{f27},~{f28},~{f29},~{f30},~{f31}"()
  %1 = load i32, i32* %ptr, align 4
  %conv = sitofp i32 %1 to float
  %2 = load float*, float** @pb, align 8
  %add.ptr = getelementptr inbounds float, float* %2, i64 16777216
  store float %conv, float* %add.ptr, align 4
  ret void
}
