; RUN: llc < %s | FileCheck --check-prefix AS %s
; RUN: opt -S -interleaved-load-combine < %s | FileCheck %s

; ModuleID = 'aarch64_interleaved-ld-combine.bc'
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "arm64--linux-gnu"

; This should be lowered into LD4
define void @aarch64_ilc_const(<4 x float>* %ptr) {
entry:

;;; Check LLVM transformation
; CHECK-LABEL: @aarch64_ilc_const(
; CHECK-DAG: [[GEP:%.+]] = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64 2
; CHECK-DAG: [[CAST:%.+]] = bitcast <4 x float>* [[GEP]] to <16 x float>*
; CHECK-DAG: [[LOAD:%.+]] = load <16 x float>, <16 x float>* [[CAST]], align 16
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
; CHECK: ret void

;;; Check if it gets lowerd
; AS-LABEL: aarch64_ilc_const
; AS: ld4
; AS: ret

  %gep1 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64  2
  %gep2 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64  3
  %gep3 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64  4
  %gep4 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64  5
  %ld1 = load <4 x float>, <4 x float>* %gep1, align 16
  %ld2 = load <4 x float>, <4 x float>* %gep2, align 16
  %ld3 = load <4 x float>, <4 x float>* %gep3, align 16
  %ld4 = load <4 x float>, <4 x float>* %gep4, align 16
  %sv1 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %sv2 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %sv3 = shufflevector <4 x float> %ld3, <4 x float> %ld4, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %sv4 = shufflevector <4 x float> %ld3, <4 x float> %ld4, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %m0_3   = shufflevector <4 x float> %sv1, <4 x float> %sv3, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m4_7   = shufflevector <4 x float> %sv1, <4 x float> %sv3, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %m8_11  = shufflevector <4 x float> %sv2, <4 x float> %sv4, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m12_15 = shufflevector <4 x float> %sv2, <4 x float> %sv4, <4 x i32> <i32 1, i32 3, i32 5, i32 7>

  store <4 x float> %m0_3, <4 x float>* %gep1, align 16
  store <4 x float> %m4_7, <4 x float>* %gep2, align 16
  store <4 x float> %m8_11, <4 x float>* %gep3, align 16
  store <4 x float> %m12_15, <4 x float>* %gep4, align 16
  ret void
}

; This should be lowered into LD4
define void @aarch64_ilc_idx(<4 x float>* %ptr, i64 %idx) {
entry:

;;; Check LLVM transformation
; CHECK-LABEL: @aarch64_ilc_idx(
; CHECK-DAG: [[ADD:%.+]] = add i64 %idx, 16
; CHECK-DAG: [[LSHR:%.+]] = lshr i64 [[ADD]], 2
; CHECK-DAG: [[GEP:%.+]] = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64 [[LSHR]]
; CHECK-DAG: [[CAST:%.+]] = bitcast <4 x float>* [[GEP]] to <16 x float>*
; CHECK-DAG: [[LOAD:%.+]] = load <16 x float>, <16 x float>* [[CAST]], align 16
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
; CHECK: ret void

; AS-LABEL: aarch64_ilc_idx
; AS-DAG: lsl [[LSL:x[0-9]+]], x1, #2
; AS-DAG: add [[ADD:x[0-9]+]], [[LSL]], #64
; AS-DAG: and [[AND:x[0-9]+]], [[ADD]], #0xfffffffffffffff0
; AS-DAG: add [[ADR:x[0-9]+]], x0, [[AND]]
; AS-DAG: ld4 { v[[V0:[0-9]+]].4s, v[[V1:[0-9]+]].4s, v[[V2:[0-9]+]].4s, v[[V3:[0-9]+]].4s }, {{\[}}[[ADR]]{{\]}}
; AS-DAG: str q[[V0]]
; AS-DAG: str q[[V1]]
; AS-DAG: str q[[V2]]
; AS-DAG: str q[[V3]]
; AS: ret

  %a2 = add i64 %idx, 20
  %idx2 = lshr i64 %a2, 2
  %a3 = add i64 %idx, 24
  %a1 = add i64 %idx, 16
  %idx1 = lshr i64 %a1, 2
  %idx3 = lshr i64 %a3, 2
  %a4 = add i64 %idx, 28
  %idx4 = lshr i64 %a4, 2

  %gep2 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64  %idx2
  %gep4 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64  %idx4
  %gep1 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64  %idx1
  %gep3 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64  %idx3
  %ld1 = load <4 x float>, <4 x float>* %gep1, align 16
  %ld2 = load <4 x float>, <4 x float>* %gep2, align 16
  %ld3 = load <4 x float>, <4 x float>* %gep3, align 16
  %ld4 = load <4 x float>, <4 x float>* %gep4, align 16
  %sv1 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %sv2 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %sv3 = shufflevector <4 x float> %ld3, <4 x float> %ld4, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %sv4 = shufflevector <4 x float> %ld3, <4 x float> %ld4, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %m0_3   = shufflevector <4 x float> %sv1, <4 x float> %sv3, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m4_7   = shufflevector <4 x float> %sv1, <4 x float> %sv3, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %m8_11  = shufflevector <4 x float> %sv2, <4 x float> %sv4, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m12_15 = shufflevector <4 x float> %sv2, <4 x float> %sv4, <4 x i32> <i32 1, i32 3, i32 5, i32 7>

  store <4 x float> %m0_3, <4 x float>* %gep1, align 16
  store <4 x float> %m4_7, <4 x float>* %gep2, align 16
  store <4 x float> %m8_11, <4 x float>* %gep3, align 16
  store <4 x float> %m12_15, <4 x float>* %gep4, align 16
  ret void
}

; This should be lowered into LD4, a offset of has to be taken into account
%struct.ilc = type <{ float, [0 x <4 x float>] }>
define void @aarch64_ilc_struct(%struct.ilc* %ptr, i64 %idx) {
entry:

;;; Check LLVM transformation
; CHECK-LABEL: @aarch64_ilc_struct(
; CHECK-DAG: [[LSHR:%.+]] = lshr i64 %idx, 2
; CHECK-DAG: [[GEP:%.+]] = getelementptr %struct.ilc, %struct.ilc* %ptr, i32 0, i32 1, i64 [[LSHR]]
; CHECK-DAG: [[CAST:%.+]] = bitcast <4 x float>* [[GEP]] to <16 x float>*
; CHECK-DAG: [[LOAD:%.+]] = load <16 x float>, <16 x float>* [[CAST]], align 4
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
; CHECK-DAG: %{{.* }}= shufflevector <16 x float> [[LOAD]], <16 x float> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
; CHECK: ret void

; AS-LABEL: aarch64_ilc_struct
; AS-DAG: lsl [[LSL:x[0-9]+]], x1, #2
; AS-DAG: add [[ADD:x[0-9]+]], x0, #4
; AS-DAG: and [[AND:x[0-9]+]], [[LSL]], #0xfffffffffffffff0
; AS-DAG: add [[ADR:x[0-9]+]], [[ADD]], [[AND]]
; AS-DAG: ld4 { v[[V0:[0-9]+]].4s, v[[V1:[0-9]+]].4s, v[[V2:[0-9]+]].4s, v[[V3:[0-9]+]].4s }, {{\[}}[[ADR]]{{\]}}
; AS-DAG: str q[[V0]]
; AS-DAG: str q[[V1]]
; AS-DAG: str q[[V2]]
; AS-DAG: str q[[V3]]
; AS: ret

  %a1 = add i64 %idx, 4
  %idx2 = lshr i64 %a1, 2
  %a2 = add i64 %idx, 8
  %idx3 = lshr i64 %a2, 2
  %a3 = add i64 %idx, 12
  %idx4 = lshr i64 %a3, 2

  %gep2 = getelementptr %struct.ilc, %struct.ilc* %ptr, i32 0, i32 1, i64 %idx2
  %gep3 = getelementptr %struct.ilc, %struct.ilc* %ptr, i32 0, i32 1, i64 %idx3
  %gep4 = getelementptr %struct.ilc, %struct.ilc* %ptr, i32 0, i32 1, i64 %idx4
  %idx1 = lshr i64 %idx, 2
  %gep1 = getelementptr %struct.ilc, %struct.ilc* %ptr, i32 0, i32 1, i64 %idx1
  %ld1 = load <4 x float>, <4 x float>* %gep1, align 4
  %ld2 = load <4 x float>, <4 x float>* %gep2, align 4
  %ld3 = load <4 x float>, <4 x float>* %gep3, align 4
  %ld4 = load <4 x float>, <4 x float>* %gep4, align 4
  %sv1 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %sv2 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %sv3 = shufflevector <4 x float> %ld3, <4 x float> %ld4, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %sv4 = shufflevector <4 x float> %ld3, <4 x float> %ld4, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %m0_3   = shufflevector <4 x float> %sv1, <4 x float> %sv3, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m4_7   = shufflevector <4 x float> %sv1, <4 x float> %sv3, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %m8_11  = shufflevector <4 x float> %sv2, <4 x float> %sv4, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m12_15 = shufflevector <4 x float> %sv2, <4 x float> %sv4, <4 x i32> <i32 1, i32 3, i32 5, i32 7>

  store <4 x float> %m0_3, <4 x float>* %gep1, align 16
  store <4 x float> %m4_7, <4 x float>* %gep2, align 16
  store <4 x float> %m8_11, <4 x float>* %gep3, align 16
  store <4 x float> %m12_15, <4 x float>* %gep4, align 16
  ret void
}

; This should be lowered into LD2
define void @aarch64_ilc_idx_ld2(<4 x float>* %ptr, i64 %idx) {
entry:
; CHECK-LABEL: @aarch64_ilc_idx_ld2(
; CHECK-DAG: [[LSHR:%.+]] = lshr i64 %idx, 2
; CHECK-DAG: [[GEP:%.+]] = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64 [[LSHR]]
; CHECK-DAG: [[CAST:%.+]] = bitcast <4 x float>* [[GEP]] to <8 x float>*
; CHECK-DAG: [[LOAD:%.+]] = load <8 x float>, <8 x float>* [[CAST]], align 16
; CHECK: %{{.* }}= shufflevector <8 x float> [[LOAD]], <8 x float> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK: %{{.* }}= shufflevector <8 x float> [[LOAD]], <8 x float> undef, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK-DAG: ret void

; AS-LABEL: aarch64_ilc_idx_ld2
; AS: ld2
; AS: ret

  %idx1 = lshr i64 %idx, 2
  %a1 = add i64 %idx, 4
  %idx2 = lshr i64 %a1, 2

  %gep1 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64  %idx1
  %gep2 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64  %idx2
  %ld1 = load <4 x float>, <4 x float>* %gep1, align 16
  %ld2 = load <4 x float>, <4 x float>* %gep2, align 16
  %m0_3 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m4_7 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 1, i32 3, i32 5, i32 7>

  store <4 x float> %m0_3, <4 x float>* %gep1
  store <4 x float> %m4_7, <4 x float>* %gep2
  ret void
}

; This should be lowered into LD3
define void @aarch64_ilc_idx_ld3(<4 x float>* %ptr, i64 %idx) {
entry:
; CHECK-LABEL: @aarch64_ilc_idx_ld3(
; CHECK-DAG: [[LSHR:%.+]] = lshr i64 %idx, 2
; CHECK-DAG: [[GEP:%.+]] = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64 [[LSHR]]
; CHECK-DAG: [[CAST:%.+]] = bitcast <4 x float>* [[GEP]] to <12 x float>*
; CHECK-DAG: [[LOAD:%.+]] = load <12 x float>, <12 x float>* [[CAST]], align 16
; CHECK: %{{.* }}= shufflevector <12 x float> [[LOAD]], <12 x float> undef, <4 x i32> <i32 0, i32 3, i32 6, i32 9>
; CHECK: %{{.* }}= shufflevector <12 x float> [[LOAD]], <12 x float> undef, <4 x i32> <i32 1, i32 4, i32 7, i32 10>
; CHECK: %{{.* }}= shufflevector <12 x float> [[LOAD]], <12 x float> undef, <4 x i32> <i32 2, i32 5, i32 8, i32 11>
; CHECK-DAG: ret void

; AS-LABEL: aarch64_ilc_idx_ld3
; AS: ld3
; AS: ret

  %idx1 = lshr i64 %idx, 2
  %a1 = add i64 %idx, 4
  %idx2 = lshr i64 %a1, 2
  %a2 = add i64 %idx, 8
  %idx3 = lshr i64 %a2, 2

  %gep1 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64  %idx1
  %gep2 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64  %idx2
  %gep3 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64  %idx3
  %ld1 = load <4 x float>, <4 x float>* %gep1, align 16
  %ld2 = load <4 x float>, <4 x float>* %gep2, align 16
  %ld3 = load <4 x float>, <4 x float>* %gep3, align 16

  %sv1 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 3, i32 6, i32 undef>
  %sv2 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 1, i32 4, i32 7, i32 undef>
  %sv3 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 2, i32 5, i32 undef, i32 undef>
  %m0_3 = shufflevector <4 x float> %sv1, <4 x float> %ld3, <4 x i32> <i32 0, i32 1, i32 2, i32 5>
  %m4_7 = shufflevector <4 x float> %sv2, <4 x float> %ld3, <4 x i32> <i32 0, i32 1, i32 2, i32 6>
  %m8_11 = shufflevector <4 x float> %sv3, <4 x float> %ld3, <4 x i32> <i32 0, i32 1, i32 4, i32 7>

  store <4 x float> %m0_3, <4 x float>* %gep1, align 16
  store <4 x float> %m4_7, <4 x float>* %gep2, align 16
  store <4 x float> %m8_11, <4 x float>* %gep3, align 16
  ret void
}
;  %sv3 = shufflevector <4 x float> %ld3, <4 x float> %ld4, <4 x i32> <i32 0, i32 undef, i32 4, i32 undef>

; This must not be lowered
define void @aarch64_ilc_i32_idx(<4 x float>* %ptr, i32 %idx) {
; CHECK-LABEL: @aarch64_ilc_i32_idx(
; CHECK: %idx1 = lshr i32 %idx, 2
; CHECK-NEXT: %a1 = add i32 %idx, 4
; CHECK-NEXT: %idx2 = lshr i32 %a1, 2
; CHECK-NEXT: %gep1 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i32 %idx1
; CHECK-NEXT: %gep2 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i32 %idx2
; CHECK-NEXT: %ld1 = load <4 x float>, <4 x float>* %gep1, align 16
; CHECK-NEXT: %ld2 = load <4 x float>, <4 x float>* %gep2, align 16
; CHECK-NEXT: %m0_3 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK-NEXT: %m4_7 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK-NEXT: store <4 x float> %m0_3, <4 x float>* %gep1, align 16
; CHECK-NEXT: store <4 x float> %m4_7, <4 x float>* %gep2, align 16
; CHECK-NEXT: ret void

; AS-LABEL: aarch64_ilc_i32_idx
; AS-DAG: @function
; AS-NOT: ld2
; AS-NOT: ld3
; AS-NOT: ld4
; AS-DAG: ret

entry:
  %idx1 = lshr i32 %idx, 2
  %a1 = add i32 %idx, 4
  %idx2 = lshr i32 %a1, 2

  %gep1 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i32 %idx1
  %gep2 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i32 %idx2
  %ld1 = load <4 x float>, <4 x float>* %gep1, align 16
  %ld2 = load <4 x float>, <4 x float>* %gep2, align 16
  %m0_3 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m4_7 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 1, i32 3, i32 5, i32 7>

  store <4 x float> %m0_3, <4 x float>* %gep1, align 16
  store <4 x float> %m4_7, <4 x float>* %gep2, align 16
  ret void
}

; Volatile loads must not be lowered
define void @aarch64_ilc_volatile(<4 x float>* %ptr) {
; CHECK-LABEL: @aarch64_ilc_volatile(
; CHECK: %gep1 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i32 0
; CHECK-NEXT: %gep2 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i32 1
; CHECK-NEXT: %ld1 = load volatile <4 x float>, <4 x float>* %gep1, align 16
; CHECK-NEXT: %ld2 = load <4 x float>, <4 x float>* %gep2, align 16
; CHECK-NEXT: %m0_3 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK-NEXT: %m4_7 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK-NEXT: store <4 x float> %m0_3, <4 x float>* %gep1, align 16
; CHECK-NEXT: store <4 x float> %m4_7, <4 x float>* %gep2, align 16
; CHECK-NEXT: ret void

; AS-LABEL: aarch64_ilc_volatile
; AS-DAG: @function
; AS-NOT: ld2
; AS-NOT: ld3
; AS-NOT: ld4
; AS-DAG: ret

entry:
  %gep1 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i32 0
  %gep2 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i32 1
  %ld1 = load volatile <4 x float>, <4 x float>* %gep1, align 16
  %ld2 = load <4 x float>, <4 x float>* %gep2, align 16
  %m0_3 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m4_7 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  store <4 x float> %m0_3, <4 x float>* %gep1, align 16
  store <4 x float> %m4_7, <4 x float>* %gep2, align 16
  ret void
}

; This must not be lowered
define void @aarch64_ilc_depmem(<4 x float>* %ptr, i32 %idx) {
entry:
; CHECK-LABEL: @aarch64_ilc_depmem(
; CHECK: %gep1 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i32 0
; CHECK-NEXT: %gep2 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i32 1
; CHECK-NEXT: %ld1 = load <4 x float>, <4 x float>* %gep1, align 16
; CHECK-NEXT: store <4 x float> %ld1, <4 x float>* %gep2, align 16
; CHECK-NEXT: %ld2 = load <4 x float>, <4 x float>* %gep2, align 16
; CHECK-NEXT: %m0_3 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK-NEXT: %m4_7 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK-NEXT: store <4 x float> %m0_3, <4 x float>* %gep1, align 16
; CHECK-NEXT: store <4 x float> %m4_7, <4 x float>* %gep2, align 16
; CHECK-NEXT: ret void

; AS-LABEL: aarch64_ilc_depmem
; AS-DAG: @function
; AS-NOT: ld2
; AS-NOT: ld3
; AS-NOT: ld4
; AS-DAG: ret

  %gep1 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i32 0
  %gep2 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i32 1
  %ld1 = load <4 x float>, <4 x float>* %gep1, align 16
  store <4 x float> %ld1, <4 x float>* %gep2, align 16
  %ld2 = load <4 x float>, <4 x float>* %gep2, align 16
  %m0_3 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %m4_7 = shufflevector <4 x float> %ld1, <4 x float> %ld2, <4 x i32> <i32 1, i32 3, i32 5, i32 7>

  store <4 x float> %m0_3, <4 x float>* %gep1, align 16
  store <4 x float> %m4_7, <4 x float>* %gep2, align 16
  ret void
}

; This cannot be converted - insertion position cannot be determined
define void @aarch64_no_insertion_pos(float* %ptr) {
entry:
; CHECK-LABEL: @aarch64_no_insertion_pos(
; CHECK: %p0 = getelementptr inbounds float, float* %ptr, i32 0
; CHECK-NEXT: %p1 = getelementptr inbounds float, float* %ptr, i32 4
; CHECK-NEXT: %b0 = bitcast float* %p0 to <5 x float>*
; CHECK-NEXT: %b1 = bitcast float* %p1 to <5 x float>*
; CHECK-NEXT: %l0 = load <5 x float>, <5 x float>* %b0
; CHECK-NEXT: %l1 = load <5 x float>, <5 x float>* %b1
; CHECK-NEXT: %s0 = shufflevector <5 x float> %l0, <5 x float> %l1, <4 x i32> <i32 1, i32 3, i32 6, i32 8>
; CHECK-NEXT: %s1 = shufflevector <5 x float> %l0, <5 x float> %l1, <4 x i32> <i32 2, i32 4, i32 7, i32 9>
; CHECK-NEXT: ret void

  %p0 = getelementptr inbounds float, float* %ptr, i32 0
  %p1 = getelementptr inbounds float, float* %ptr, i32 4
  %b0 = bitcast float* %p0 to <5 x float>*
  %b1 = bitcast float* %p1 to <5 x float>*
  %l0 = load <5 x float>, <5 x float>* %b0
  %l1 = load <5 x float>, <5 x float>* %b1
  %s0 = shufflevector <5 x float> %l0, <5 x float> %l1, <4 x i32> <i32 1, i32 3, i32 6, i32 8>
  %s1 = shufflevector <5 x float> %l0, <5 x float> %l1, <4 x i32> <i32 2, i32 4, i32 7, i32 9>
  ret void
}

; This cannot be converted - the insertion position does not dominate all
; uses
define void @aarch64_insertpos_does_not_dominate(float* %ptr) {
entry:
; CHECK-LABEL: @aarch64_insertpos_does_not_dominate(
; CHECK: %p0 = getelementptr inbounds float, float* %ptr, i32 0
; CHECK-NEXT: %p1 = getelementptr inbounds float, float* %ptr, i32 1
; CHECK-NEXT: %b0 = bitcast float* %p0 to <7 x float>*
; CHECK-NEXT: %b1 = bitcast float* %p1 to <7 x float>*
; CHECK-NEXT: %l1 = load <7 x float>, <7 x float>* %b1
; CHECK-NEXT: %s1 = shufflevector <7 x float> %l1, <7 x float> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK-NEXT: %l0 = load <7 x float>, <7 x float>* %b0
; CHECK-NEXT: %s0 = shufflevector <7 x float> %l0, <7 x float> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK-NEXT: ret void
  %p0 = getelementptr inbounds float, float* %ptr, i32 0
  %p1 = getelementptr inbounds float, float* %ptr, i32 1
  %b0 = bitcast float* %p0 to <7 x float>*
  %b1 = bitcast float* %p1 to <7 x float>*
  %l1 = load <7 x float>, <7 x float>* %b1
  %s1 = shufflevector <7 x float> %l1, <7 x float> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %l0 = load <7 x float>, <7 x float>* %b0
  %s0 = shufflevector <7 x float> %l0, <7 x float> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  ret void
}
