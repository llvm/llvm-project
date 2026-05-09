; RUN: llc -mtriple=armv7-eabi -mattr=+neon -float-abi=hard < %s | FileCheck --check-prefix=AS %s
; RUN: opt -S -mtriple=armv7-eabi -mattr=+neon -interleaved-load-combine < %s | FileCheck %s
; RUN: opt -S -mtriple=armv7-eabi -mattr=+neon -passes=interleaved-load-combine < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"
target triple = "armv7---eabi"

; This should be lowered into VLD4.32
define void @arm_ilc_const(ptr %ptr) {
entry:

;;; Check LLVM transformation
; CHECK-LABEL: @arm_ilc_const(
; CHECK-DAG: [[GEP:%.+]] = getelementptr inbounds <4 x float>, ptr %ptr, i32 2
; CHECK-DAG: [[LOAD:%.+]] = load <16 x float>, ptr [[GEP]], align 16
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> poison, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> poison, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> poison, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> poison, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
; CHECK: ret void

;;; Check if it gets lowered
; AS-LABEL: arm_ilc_const
; AS: vld4.32
; AS: bx lr

  %gep1 = getelementptr inbounds <4 x float>, ptr %ptr, i32 2
  %gep2 = getelementptr inbounds <4 x float>, ptr %ptr, i32 3
  %gep3 = getelementptr inbounds <4 x float>, ptr %ptr, i32 4
  %gep4 = getelementptr inbounds <4 x float>, ptr %ptr, i32 5
  %ld1 = load <4 x float>, ptr %gep1, align 16
  %ld2 = load <4 x float>, ptr %gep2, align 16
  %ld3 = load <4 x float>, ptr %gep3, align 16
  %ld4 = load <4 x float>, ptr %gep4, align 16
  %sv1 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %sv2 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %sv3 = shufflevector <4 x float> %ld3, <4 x float> %ld4, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %sv4 = shufflevector <4 x float> %ld3, <4 x float> %ld4, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %m0_3   = shufflevector <4 x float> %sv1, <4 x float> %sv3, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m4_7   = shufflevector <4 x float> %sv1, <4 x float> %sv3, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %m8_11  = shufflevector <4 x float> %sv2, <4 x float> %sv4, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m12_15 = shufflevector <4 x float> %sv2, <4 x float> %sv4, <4 x i32> <i32 1, i32 3, i32 5, i32 7>

  store <4 x float> %m0_3, ptr %gep1, align 16
  store <4 x float> %m4_7, ptr %gep2, align 16
  store <4 x float> %m8_11, ptr %gep3, align 16
  store <4 x float> %m12_15, ptr %gep4, align 16
  ret void
}

; This should be lowered into VLD4.32
define void @arm_ilc_idx(ptr %ptr, i32 %idx) {
entry:

;;; Check LLVM transformation
; CHECK-LABEL: @arm_ilc_idx(
; CHECK-DAG: [[ADD:%.+]] = add i32 %idx, 16
; CHECK-DAG: [[LSHR:%.+]] = lshr i32 [[ADD]], 2
; CHECK-DAG: [[GEP:%.+]] = getelementptr inbounds <4 x float>, ptr %ptr, i32 [[LSHR]]
; CHECK-DAG: [[LOAD:%.+]] = load <16 x float>, ptr [[GEP]], align 16
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> poison, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> poison, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> poison, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> poison, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
; CHECK: ret void

; AS-LABEL: arm_ilc_idx
; AS: vld4.32
; AS: bx lr

  %a2 = add i32 %idx, 20
  %idx2 = lshr i32 %a2, 2
  %a3 = add i32 %idx, 24
  %a1 = add i32 %idx, 16
  %idx1 = lshr i32 %a1, 2
  %idx3 = lshr i32 %a3, 2
  %a4 = add i32 %idx, 28
  %idx4 = lshr i32 %a4, 2

  %gep2 = getelementptr inbounds <4 x float>, ptr %ptr, i32 %idx2
  %gep4 = getelementptr inbounds <4 x float>, ptr %ptr, i32 %idx4
  %gep1 = getelementptr inbounds <4 x float>, ptr %ptr, i32 %idx1
  %gep3 = getelementptr inbounds <4 x float>, ptr %ptr, i32 %idx3
  %ld1 = load <4 x float>, ptr %gep1, align 16
  %ld2 = load <4 x float>, ptr %gep2, align 16
  %ld3 = load <4 x float>, ptr %gep3, align 16
  %ld4 = load <4 x float>, ptr %gep4, align 16
  %sv1 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %sv2 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %sv3 = shufflevector <4 x float> %ld3, <4 x float> %ld4, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %sv4 = shufflevector <4 x float> %ld3, <4 x float> %ld4, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %m0_3   = shufflevector <4 x float> %sv1, <4 x float> %sv3, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m4_7   = shufflevector <4 x float> %sv1, <4 x float> %sv3, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %m8_11  = shufflevector <4 x float> %sv2, <4 x float> %sv4, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m12_15 = shufflevector <4 x float> %sv2, <4 x float> %sv4, <4 x i32> <i32 1, i32 3, i32 5, i32 7>

  store <4 x float> %m0_3, ptr %gep1, align 16
  store <4 x float> %m4_7, ptr %gep2, align 16
  store <4 x float> %m8_11, ptr %gep3, align 16
  store <4 x float> %m12_15, ptr %gep4, align 16
  ret void
}

; This should be lowered into VLD4.32, an offset has to be taken into account
%struct.ilc = type <{ float, [0 x <4 x float>] }>
define void @arm_ilc_struct(ptr %ptr, i32 %idx) {
entry:

;;; Check LLVM transformation
; CHECK-LABEL: @arm_ilc_struct(
; CHECK-DAG: [[LSHR:%.+]] = lshr i32 %idx, 2
; CHECK-DAG: [[GEP:%.+]] = getelementptr %struct.ilc, ptr %ptr, i32 0, i32 1, i32 [[LSHR]]
; CHECK-DAG: [[LOAD:%.+]] = load <16 x float>, ptr [[GEP]], align 4
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> poison, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> poison, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> poison, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> poison, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
; CHECK: ret void

; AS-LABEL: arm_ilc_struct
; AS: vld4.32
; AS: bx lr

  %a1 = add i32 %idx, 4
  %idx2 = lshr i32 %a1, 2
  %a2 = add i32 %idx, 8
  %idx3 = lshr i32 %a2, 2
  %a3 = add i32 %idx, 12
  %idx4 = lshr i32 %a3, 2

  %gep2 = getelementptr %struct.ilc, ptr %ptr, i32 0, i32 1, i32 %idx2
  %gep3 = getelementptr %struct.ilc, ptr %ptr, i32 0, i32 1, i32 %idx3
  %gep4 = getelementptr %struct.ilc, ptr %ptr, i32 0, i32 1, i32 %idx4
  %idx1 = lshr i32 %idx, 2
  %gep1 = getelementptr %struct.ilc, ptr %ptr, i32 0, i32 1, i32 %idx1
  %ld1 = load <4 x float>, ptr %gep1, align 4
  %ld2 = load <4 x float>, ptr %gep2, align 4
  %ld3 = load <4 x float>, ptr %gep3, align 4
  %ld4 = load <4 x float>, ptr %gep4, align 4
  %sv1 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %sv2 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %sv3 = shufflevector <4 x float> %ld3, <4 x float> %ld4, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %sv4 = shufflevector <4 x float> %ld3, <4 x float> %ld4, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %m0_3   = shufflevector <4 x float> %sv1, <4 x float> %sv3, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m4_7   = shufflevector <4 x float> %sv1, <4 x float> %sv3, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %m8_11  = shufflevector <4 x float> %sv2, <4 x float> %sv4, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m12_15 = shufflevector <4 x float> %sv2, <4 x float> %sv4, <4 x i32> <i32 1, i32 3, i32 5, i32 7>

  store <4 x float> %m0_3, ptr %gep1, align 16
  store <4 x float> %m4_7, ptr %gep2, align 16
  store <4 x float> %m8_11, ptr %gep3, align 16
  store <4 x float> %m12_15, ptr %gep4, align 16
  ret void
}

; This should be lowered into VLD2.32
define void @arm_ilc_idx_ld2(ptr %ptr, i32 %idx) {
entry:
; CHECK-LABEL: @arm_ilc_idx_ld2(
; CHECK-DAG: [[LSHR:%.+]] = lshr i32 %idx, 2
; CHECK-DAG: [[GEP:%.+]] = getelementptr inbounds <4 x float>, ptr %ptr, i32 [[LSHR]]
; CHECK-DAG: [[LOAD:%.+]] = load <8 x float>, ptr [[GEP]], align 16
; CHECK: %{{.* }}= shufflevector <8 x float> [[LOAD]], <8 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK: %{{.* }}= shufflevector <8 x float> [[LOAD]], <8 x float> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK-DAG: ret void

; AS-LABEL: arm_ilc_idx_ld2
; AS: vld2.32
; AS: bx lr

  %idx1 = lshr i32 %idx, 2
  %a1 = add i32 %idx, 4
  %idx2 = lshr i32 %a1, 2

  %gep1 = getelementptr inbounds <4 x float>, ptr %ptr, i32 %idx1
  %gep2 = getelementptr inbounds <4 x float>, ptr %ptr, i32 %idx2
  %ld1 = load <4 x float>, ptr %gep1, align 16
  %ld2 = load <4 x float>, ptr %gep2, align 16
  %m0_3 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m4_7 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 1, i32 3, i32 5, i32 7>

  store <4 x float> %m0_3, ptr %gep1
  store <4 x float> %m4_7, ptr %gep2
  ret void
}

; This should be lowered into VLD3.32
define void @arm_ilc_idx_ld3(ptr %ptr, i32 %idx) {
entry:
; CHECK-LABEL: @arm_ilc_idx_ld3(
; CHECK-DAG: [[LSHR:%.+]] = lshr i32 %idx, 2
; CHECK-DAG: [[GEP:%.+]] = getelementptr inbounds <4 x float>, ptr %ptr, i32 [[LSHR]]
; CHECK-DAG: [[LOAD:%.+]] = load <12 x float>, ptr [[GEP]], align 16
; CHECK: %{{.* }}= shufflevector <12 x float> [[LOAD]], <12 x float> poison, <4 x i32> <i32 0, i32 3, i32 6, i32 9>
; CHECK: %{{.* }}= shufflevector <12 x float> [[LOAD]], <12 x float> poison, <4 x i32> <i32 1, i32 4, i32 7, i32 10>
; CHECK: %{{.* }}= shufflevector <12 x float> [[LOAD]], <12 x float> poison, <4 x i32> <i32 2, i32 5, i32 8, i32 11>
; CHECK-DAG: ret void

; AS-LABEL: arm_ilc_idx_ld3
; AS: vld3.32
; AS: bx lr

  %idx1 = lshr i32 %idx, 2
  %a1 = add i32 %idx, 4
  %idx2 = lshr i32 %a1, 2
  %a2 = add i32 %idx, 8
  %idx3 = lshr i32 %a2, 2

  %gep1 = getelementptr inbounds <4 x float>, ptr %ptr, i32 %idx1
  %gep2 = getelementptr inbounds <4 x float>, ptr %ptr, i32 %idx2
  %gep3 = getelementptr inbounds <4 x float>, ptr %ptr, i32 %idx3
  %ld1 = load <4 x float>, ptr %gep1, align 16
  %ld2 = load <4 x float>, ptr %gep2, align 16
  %ld3 = load <4 x float>, ptr %gep3, align 16

  %sv1 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 3, i32 6, i32 undef>
  %sv2 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 1, i32 4, i32 7, i32 undef>
  %sv3 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 2, i32 5, i32 undef, i32 undef>
  %m0_3 = shufflevector <4 x float> %sv1, <4 x float> %ld3, <4 x i32> <i32 0, i32 1, i32 2, i32 5>
  %m4_7 = shufflevector <4 x float> %sv2, <4 x float> %ld3, <4 x i32> <i32 0, i32 1, i32 2, i32 6>
  %m8_11 = shufflevector <4 x float> %sv3, <4 x float> %ld3, <4 x i32> <i32 0, i32 1, i32 4, i32 7>

  store <4 x float> %m0_3, ptr %gep1, align 16
  store <4 x float> %m4_7, ptr %gep2, align 16
  store <4 x float> %m8_11, ptr %gep3, align 16
  ret void
}

; A wider (i64) induction variable on a 32-bit target is also handled: the
; pass tracks the low 32 bits of the index arithmetic and lowers to VLD2.32.
define void @arm_ilc_i64_idx_ld2(ptr %ptr, i64 %idx) {
entry:
; CHECK-LABEL: @arm_ilc_i64_idx_ld2(
; CHECK-DAG: [[LSHR:%.+]] = lshr i64 %idx, 2
; CHECK-DAG: [[GEP:%.+]] = getelementptr inbounds <4 x float>, ptr %ptr, i64 [[LSHR]]
; CHECK-DAG: [[LOAD:%.+]] = load <8 x float>, ptr [[GEP]], align 16
; CHECK-DAG: %{{.* }}= shufflevector <8 x float> [[LOAD]], <8 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK-DAG: %{{.* }}= shufflevector <8 x float> [[LOAD]], <8 x float> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK: ret void

; AS-LABEL: arm_ilc_i64_idx_ld2
; AS: vld2.32
; AS: bx lr

  %idx1 = lshr i64 %idx, 2
  %a1 = add i64 %idx, 4
  %idx2 = lshr i64 %a1, 2

  %gep1 = getelementptr inbounds <4 x float>, ptr %ptr, i64 %idx1
  %gep2 = getelementptr inbounds <4 x float>, ptr %ptr, i64 %idx2
  %ld1 = load <4 x float>, ptr %gep1, align 16
  %ld2 = load <4 x float>, ptr %gep2, align 16
  %m0_3 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m4_7 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 1, i32 3, i32 5, i32 7>

  store <4 x float> %m0_3, ptr %gep1, align 16
  store <4 x float> %m4_7, ptr %gep2, align 16
  ret void
}

; Volatile loads must not be lowered
define void @arm_ilc_volatile(ptr %ptr) {
; CHECK-LABEL: @arm_ilc_volatile(
; CHECK: %gep2 = getelementptr inbounds <4 x float>, ptr %ptr, i32 1
; CHECK-NEXT: %ld1 = load volatile <4 x float>, ptr %ptr, align 16
; CHECK-NEXT: %ld2 = load <4 x float>, ptr %gep2, align 16
; CHECK-NEXT: %m0_3 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK-NEXT: %m4_7 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK-NEXT: store <4 x float> %m0_3, ptr %ptr, align 16
; CHECK-NEXT: store <4 x float> %m4_7, ptr %gep2, align 16
; CHECK-NEXT: ret void

; AS-LABEL: arm_ilc_volatile
; AS-NOT: vld2.
; AS-NOT: vld3.
; AS-NOT: vld4.
; AS: bx lr

entry:
  %gep2 = getelementptr inbounds <4 x float>, ptr %ptr, i32 1
  %ld1 = load volatile <4 x float>, ptr %ptr, align 16
  %ld2 = load <4 x float>, ptr %gep2, align 16
  %m0_3 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m4_7 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  store <4 x float> %m0_3, ptr %ptr, align 16
  store <4 x float> %m4_7, ptr %gep2, align 16
  ret void
}

; This must not be lowered because there is an aliasing store between the loads
define void @arm_ilc_depmem(ptr %ptr, i32 %idx) {
entry:
; CHECK-LABEL: @arm_ilc_depmem(
; CHECK: %gep2 = getelementptr inbounds <4 x float>, ptr %ptr, i32 1
; CHECK-NEXT: %ld1 = load <4 x float>, ptr %ptr, align 16
; CHECK-NEXT: store <4 x float> %ld1, ptr %gep2, align 16
; CHECK-NEXT: %ld2 = load <4 x float>, ptr %gep2, align 16
; CHECK-NEXT: %m0_3 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK-NEXT: %m4_7 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK-NEXT: store <4 x float> %m0_3, ptr %ptr, align 16
; CHECK-NEXT: store <4 x float> %m4_7, ptr %gep2, align 16
; CHECK-NEXT: ret void

; AS-LABEL: arm_ilc_depmem
; AS-NOT: vld2.
; AS-NOT: vld3.
; AS-NOT: vld4.
; AS: bx lr

  %gep2 = getelementptr inbounds <4 x float>, ptr %ptr, i32 1
  %ld1 = load <4 x float>, ptr %ptr, align 16
  store <4 x float> %ld1, ptr %gep2, align 16
  %ld2 = load <4 x float>, ptr %gep2, align 16
  %m0_3 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m4_7 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 1, i32 3, i32 5, i32 7>

  store <4 x float> %m0_3, ptr %ptr, align 16
  store <4 x float> %m4_7, ptr %gep2, align 16
  ret void
}

; This cannot be converted - insertion position cannot be determined
define void @arm_no_insertion_pos(ptr %ptr) {
entry:
; CHECK-LABEL: @arm_no_insertion_pos(
; CHECK: %p1 = getelementptr inbounds float, ptr %ptr, i32 4
; CHECK-NEXT: %l0 = load <5 x float>, ptr %ptr
; CHECK-NEXT: %l1 = load <5 x float>, ptr %p1
; CHECK-NEXT: %s0 = shufflevector <5 x float> %l0, <5 x float> %l1, <4 x i32> <i32 1, i32 3, i32 6, i32 8>
; CHECK-NEXT: %s1 = shufflevector <5 x float> %l0, <5 x float> %l1, <4 x i32> <i32 2, i32 4, i32 7, i32 9>
; CHECK-NEXT: ret void

  %p1 = getelementptr inbounds float, ptr %ptr, i32 4
  %l0 = load <5 x float>, ptr %ptr
  %l1 = load <5 x float>, ptr %p1
  %s0 = shufflevector <5 x float> %l0, <5 x float> %l1, <4 x i32> <i32 1, i32 3, i32 6, i32 8>
  %s1 = shufflevector <5 x float> %l0, <5 x float> %l1, <4 x i32> <i32 2, i32 4, i32 7, i32 9>
  ret void
}

; This cannot be converted - the insertion position does not dominate all uses
define void @arm_insertpos_does_not_dominate(ptr %ptr) {
entry:
; CHECK-LABEL: @arm_insertpos_does_not_dominate(
; CHECK: %p1 = getelementptr inbounds float, ptr %ptr, i32 1
; CHECK-NEXT: %l1 = load <7 x float>, ptr %p1
; CHECK-NEXT: %s1 = shufflevector <7 x float> %l1, <7 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK-NEXT: %l0 = load <7 x float>, ptr %ptr
; CHECK-NEXT: %s0 = shufflevector <7 x float> %l0, <7 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK-NEXT: ret void
  %p1 = getelementptr inbounds float, ptr %ptr, i32 1
  %l1 = load <7 x float>, ptr %p1
  %s1 = shufflevector <7 x float> %l1, <7 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %l0 = load <7 x float>, ptr %ptr
  %s0 = shufflevector <7 x float> %l0, <7 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  ret void
}
