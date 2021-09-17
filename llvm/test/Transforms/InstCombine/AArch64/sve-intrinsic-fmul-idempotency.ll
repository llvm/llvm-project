; RUN: opt -S -instcombine < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

; Idempotent fmuls -- should compile to just a ret.
define <vscale x 8 x half> @idempotent_fmul_f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a) #0 {
; CHECK-LABEL: @idempotent_fmul_f16(
; CHECK-NEXT:    ret <vscale x 8 x half> [[A:%.*]]
;
  %1 = call <vscale x 8 x half> @llvm.aarch64.sve.dup.x.nxv8f16(half 1.0)
  %2 = call <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %1)
  ret <vscale x 8 x half> %2
}

define <vscale x 4 x float> @idempotent_fmul_f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a) #0 {
; CHECK-LABEL: @idempotent_fmul_f32(
; CHECK-NEXT:    ret <vscale x 4 x float> [[A:%.*]]
;
  %1 = call <vscale x 4 x float> @llvm.aarch64.sve.dup.x.nxv4f32(float 1.0)
  %2 = call <vscale x 4 x float> @llvm.aarch64.sve.fmul.nxv4f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %1)
  ret <vscale x 4 x float> %2
}

define <vscale x 2 x double> @idempotent_fmul_f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a) #0 {
; CHECK-LABEL: @idempotent_fmul_f64(
; CHECK-NEXT:    ret <vscale x 2 x double> [[A:%.*]]
;
  %1 = call <vscale x 2 x double> @llvm.aarch64.sve.dup.x.nxv2f64(double 1.0)
  %2 = call <vscale x 2 x double> @llvm.aarch64.sve.fmul.nxv2f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %1)
  ret <vscale x 2 x double> %2
}

define <vscale x 2 x double> @idempotent_fmul_different_argument_order(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a) #0 {
; CHECK-LABEL: @idempotent_fmul_different_argument_order(
; CHECK-NEXT:    ret <vscale x 2 x double> [[A:%.*]]
;
  %1 = call <vscale x 2 x double> @llvm.aarch64.sve.dup.x.nxv2f64(double 1.0)
  ; Different argument order to the above tests.
  %2 = call <vscale x 2 x double> @llvm.aarch64.sve.fmul.nxv2f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %1, <vscale x 2 x double> %a)
  ret <vscale x 2 x double> %2
}

define <vscale x 8 x half> @idempotent_fmul_with_predicated_dup(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a) #0 {
; CHECK-LABEL: @idempotent_fmul_with_predicated_dup(
; CHECK-NEXT:    ret <vscale x 8 x half> [[A:%.*]]
;
  %1 = call <vscale x 8 x half> @llvm.aarch64.sve.dup.nxv8f16(<vscale x 8 x half> undef, <vscale x 8 x i1> %pg, half 1.0)
  %2 = call <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %1)
  ret <vscale x 8 x half> %2
}

define <vscale x 8 x half> @idempotent_fmul_two_dups(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a) #0 {
  ; Edge case -- make sure that the case where we're fmultiplying two dups
  ; together is sane.
; CHECK-LABEL: @idempotent_fmul_two_dups(
; CHECK-NEXT:    ret <vscale x 8 x half> shufflevector (<vscale x 8 x half> insertelement (<vscale x 8 x half> poison, half 0xH3C00, i32 0), <vscale x 8 x half> poison, <vscale x 8 x i32> zeroinitializer)
;
  %1 = call <vscale x 8 x half> @llvm.aarch64.sve.dup.x.nxv8f16(half 1.0)
  %2 = call <vscale x 8 x half> @llvm.aarch64.sve.dup.x.nxv8f16(half 1.0)
  %3 = call <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %1, <vscale x 8 x half> %2)
  ret <vscale x 8 x half> %3
}

; Non-idempotent fmuls -- we don't expect these to be optimised out.
define <vscale x 8 x half> @non_idempotent_fmul_f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a) #0 {
; CHECK-LABEL: @non_idempotent_fmul_f16(
; CHECK-NEXT:    [[TMP1:%.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> [[PG:%.*]], <vscale x 8 x half> [[A:%.*]], <vscale x 8 x half> shufflevector (<vscale x 8 x half> insertelement (<vscale x 8 x half> poison, half 0xH4000, i32 0), <vscale x 8 x half> poison, <vscale x 8 x i32> zeroinitializer))
; CHECK-NEXT:    ret <vscale x 8 x half> [[TMP1]]
;
  %1 = call <vscale x 8 x half> @llvm.aarch64.sve.dup.x.nxv8f16(half 2.0)
  %2 = call <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %pg, <vscale x 8 x half> %a, <vscale x 8 x half> %1)
  ret <vscale x 8 x half> %2
}

define <vscale x 4 x float> @non_idempotent_fmul_f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a) #0 {
; CHECK-LABEL: @non_idempotent_fmul_f32(
; CHECK-NEXT:    [[TMP1:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fmul.nxv4f32(<vscale x 4 x i1> [[PG:%.*]], <vscale x 4 x float> [[A:%.*]], <vscale x 4 x float> shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float 2.000000e+00, i32 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer))
; CHECK-NEXT:    ret <vscale x 4 x float> [[TMP1]]
;
  %1 = call <vscale x 4 x float> @llvm.aarch64.sve.dup.x.nxv4f32(float 2.0)
  %2 = call <vscale x 4 x float> @llvm.aarch64.sve.fmul.nxv4f32(<vscale x 4 x i1> %pg, <vscale x 4 x float> %a, <vscale x 4 x float> %1)
  ret <vscale x 4 x float> %2
}

define <vscale x 2 x double> @non_idempotent_fmul_f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a) #0 {
; CHECK-LABEL: @non_idempotent_fmul_f64(
; CHECK-NEXT:    [[TMP1:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fmul.nxv2f64(<vscale x 2 x i1> [[PG:%.*]], <vscale x 2 x double> [[A:%.*]], <vscale x 2 x double> shufflevector (<vscale x 2 x double> insertelement (<vscale x 2 x double> poison, double 2.000000e+00, i32 0), <vscale x 2 x double> poison, <vscale x 2 x i32> zeroinitializer))
; CHECK-NEXT:    ret <vscale x 2 x double> [[TMP1]]
;
  %1 = call <vscale x 2 x double> @llvm.aarch64.sve.dup.x.nxv2f64(double 2.0)
  %2 = call <vscale x 2 x double> @llvm.aarch64.sve.fmul.nxv2f64(<vscale x 2 x i1> %pg, <vscale x 2 x double> %a, <vscale x 2 x double> %1)
  ret <vscale x 2 x double> %2
}

define <vscale x 2 x double> @non_idempotent_fmul_with_predicated_dup(<vscale x 2 x i1> %pg1, <vscale x 2 x i1> %pg2, <vscale x 2 x double> %a) #0 {
  ; Different predicates
; CHECK-LABEL: @non_idempotent_fmul_with_predicated_dup(
; CHECK-NEXT:    [[TMP1:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.dup.nxv2f64(<vscale x 2 x double> undef, <vscale x 2 x i1> [[PG1:%.*]], double 1.000000e+00)
; CHECK-NEXT:    [[TMP2:%.*]] = call <vscale x 2 x double> @llvm.aarch64.sve.fmul.nxv2f64(<vscale x 2 x i1> [[PG2:%.*]], <vscale x 2 x double> [[A:%.*]], <vscale x 2 x double> [[TMP1]])
; CHECK-NEXT:    ret <vscale x 2 x double> [[TMP2]]
;
  %1 = call <vscale x 2 x double> @llvm.aarch64.sve.dup.nxv2f64(<vscale x 2 x double> undef, <vscale x 2 x i1> %pg1, double 1.0)
  %2 = call <vscale x 2 x double> @llvm.aarch64.sve.fmul.nxv2f64(<vscale x 2 x i1> %pg2, <vscale x 2 x double> %a, <vscale x 2 x double> %1)
  ret <vscale x 2 x double> %2
}

declare <vscale x 8 x half> @llvm.aarch64.sve.dup.x.nxv8f16(half)
declare <vscale x 4 x float> @llvm.aarch64.sve.dup.x.nxv4f32(float)
declare <vscale x 2 x double> @llvm.aarch64.sve.dup.x.nxv2f64(double)

declare <vscale x 2 x double> @llvm.aarch64.sve.dup.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, double)
declare <vscale x 8 x half> @llvm.aarch64.sve.dup.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, half)

declare <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>, <vscale x 8 x half>)
declare <vscale x 4 x float> @llvm.aarch64.sve.fmul.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmul.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

attributes #0 = { "target-features"="+sve" }
