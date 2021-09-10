; RUN: opt -S -instcombine < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

; Idempotent muls -- should compile to just a ret.
define <vscale x 8 x i16> @idempotent_mul_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) #0 {
; CHECK-LABEL: @idempotent_mul_i16(
; CHECK-NEXT:    ret <vscale x 8 x i16> [[A:%.*]]
;
  %1 = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 1)
  %2 = call <vscale x 8 x i16> @llvm.aarch64.sve.mul.nxv8i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %1)
  ret <vscale x 8 x i16> %2
}

define <vscale x 4 x i32> @idempotent_mul_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) #0 {
; CHECK-LABEL: @idempotent_mul_i32(
; CHECK-NEXT:    ret <vscale x 4 x i32> [[A:%.*]]
;
  %1 = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 1)
  %2 = call <vscale x 4 x i32> @llvm.aarch64.sve.mul.nxv4i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %1)
  ret <vscale x 4 x i32> %2
}

define <vscale x 2 x i64> @idempotent_mul_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) #0 {
; CHECK-LABEL: @idempotent_mul_i64(
; CHECK-NEXT:    ret <vscale x 2 x i64> [[A:%.*]]
;
  %1 = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 1)
  %2 = call <vscale x 2 x i64> @llvm.aarch64.sve.mul.nxv2i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %1)
  ret <vscale x 2 x i64> %2
}

define <vscale x 2 x i64> @idempotent_mul_different_argument_order(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) #0 {
; CHECK-LABEL: @idempotent_mul_different_argument_order(
; CHECK-NEXT:    ret <vscale x 2 x i64> [[A:%.*]]
;
  %1 = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 1)
  ; Different argument order to the above tests.
  %2 = call <vscale x 2 x i64> @llvm.aarch64.sve.mul.nxv2i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %1, <vscale x 2 x i64> %a)
  ret <vscale x 2 x i64> %2
}

define <vscale x 8 x i16> @idempotent_mul_with_predicated_dup(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) #0 {
; CHECK-LABEL: @idempotent_mul_with_predicated_dup(
; CHECK-NEXT:    ret <vscale x 8 x i16> [[A:%.*]]
;
  %1 = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.nxv8i16(<vscale x 8 x i16> undef, <vscale x 8 x i1> %pg, i16 1)
  %2 = call <vscale x 8 x i16> @llvm.aarch64.sve.mul.nxv8i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %1)
  ret <vscale x 8 x i16> %2
}

define <vscale x 8 x i16> @idempotent_mul_two_dups(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) #0 {
  ; Edge case -- make sure that the case where we're multiplying two dups
  ; together is sane.
; CHECK-LABEL: @idempotent_mul_two_dups(
; CHECK-NEXT:    [[TMP1:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 1)
; CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP1]]
;
  %1 = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 1)
  %2 = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 1)
  %3 = call <vscale x 8 x i16> @llvm.aarch64.sve.mul.nxv8i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %1, <vscale x 8 x i16> %2)
  ret <vscale x 8 x i16> %3
}

; Non-idempotent muls -- we don't expect these to be optimised out.
define <vscale x 8 x i16> @non_idempotent_mul_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) #0 {
; CHECK-LABEL: @non_idempotent_mul_i16(
; CHECK-NEXT:    [[TMP1:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 2)
; CHECK-NEXT:    [[TMP2:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.mul.nxv8i16(<vscale x 8 x i1> [[PG:%.*]], <vscale x 8 x i16> [[A:%.*]], <vscale x 8 x i16> [[TMP1]])
; CHECK-NEXT:    ret <vscale x 8 x i16> [[TMP2]]
;
  %1 = call <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16 2)
  %2 = call <vscale x 8 x i16> @llvm.aarch64.sve.mul.nxv8i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %1)
  ret <vscale x 8 x i16> %2
}

define <vscale x 4 x i32> @non_idempotent_mul_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) #0 {
; CHECK-LABEL: @non_idempotent_mul_i32(
; CHECK-NEXT:    [[TMP1:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 2)
; CHECK-NEXT:    [[TMP2:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.mul.nxv4i32(<vscale x 4 x i1> [[PG:%.*]], <vscale x 4 x i32> [[A:%.*]], <vscale x 4 x i32> [[TMP1]])
; CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP2]]
;
  %1 = call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 2)
  %2 = call <vscale x 4 x i32> @llvm.aarch64.sve.mul.nxv4i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %1)
  ret <vscale x 4 x i32> %2
}

define <vscale x 2 x i64> @non_idempotent_mul_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) #0 {
; CHECK-LABEL: @non_idempotent_mul_i64(
; CHECK-NEXT:    [[TMP1:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 2)
; CHECK-NEXT:    [[TMP2:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.mul.nxv2i64(<vscale x 2 x i1> [[PG:%.*]], <vscale x 2 x i64> [[A:%.*]], <vscale x 2 x i64> [[TMP1]])
; CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP2]]
;
  %1 = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 2)
  %2 = call <vscale x 2 x i64> @llvm.aarch64.sve.mul.nxv2i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %1)
  ret <vscale x 2 x i64> %2
}

define <vscale x 2 x i64> @non_idempotent_mul_with_predicated_dup(<vscale x 2 x i1> %pg1, <vscale x 2 x i1> %pg2, <vscale x 2 x i64> %a) #0 {
  ; Different predicates
; CHECK-LABEL: @non_idempotent_mul_with_predicated_dup(
; CHECK-NEXT:    [[TMP1:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.nxv2i64(<vscale x 2 x i64> undef, <vscale x 2 x i1> [[PG1:%.*]], i64 1)
; CHECK-NEXT:    [[TMP2:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.mul.nxv2i64(<vscale x 2 x i1> [[PG2:%.*]], <vscale x 2 x i64> [[A:%.*]], <vscale x 2 x i64> [[TMP1]])
; CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP2]]
;
  %1 = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.nxv2i64(<vscale x 2 x i64> undef, <vscale x 2 x i1> %pg1, i64 1)
  %2 = call <vscale x 2 x i64> @llvm.aarch64.sve.mul.nxv2i64(<vscale x 2 x i1> %pg2, <vscale x 2 x i64> %a, <vscale x 2 x i64> %1)
  ret <vscale x 2 x i64> %2
}

declare <vscale x 8 x i16> @llvm.aarch64.sve.dup.x.nxv8i16(i16)
declare <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64)

declare <vscale x 2 x i64> @llvm.aarch64.sve.dup.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, i64)
declare <vscale x 8 x i16> @llvm.aarch64.sve.dup.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, i16)

declare <vscale x 8 x i16> @llvm.aarch64.sve.mul.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.mul.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.mul.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

attributes #0 = { "target-features"="+sve" }
