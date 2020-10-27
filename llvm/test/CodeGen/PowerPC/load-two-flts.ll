; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"

define void @_Z4testSt7complexIfE(float %v0, float %v1, i64* %ref.tmp, float* %_M_value.realp.i.i, float* %_M_value.imagp.i.i) {
entry:
  %v2 = load i64, i64* %ref.tmp, align 8
  %v3 = lshr i64 %v2, 32
  %v4 = trunc i64 %v3 to i32
  %v5 = bitcast i32 %v4 to float
  %v6 = trunc i64 %v2 to i32
  %v7 = bitcast i32 %v6 to float
  %mul_ad.i.i = fmul float %v5, %v1
  %mul_bc.i.i = fmul float %v7, %v0
  %mul_i.i.i = fadd float %mul_ad.i.i, %mul_bc.i.i
  %mul_ac.i.i = fmul float %v5, %v0
  %mul_bd.i.i = fmul float %v7, %v1
  %mul_r.i.i = fsub float %mul_ac.i.i, %mul_bd.i.i
  store float %mul_r.i.i, float* %_M_value.realp.i.i, align 4
  store float %mul_i.i.i, float* %_M_value.imagp.i.i, align 4
  ret void

; CHECK-LABEL: @_Z4testSt7complexIfE
; CHECK-NOT: ld {{[0-9]+}}, 0(5)
; CHECK-NOT: stw
; CHECK-NOT: rldicl
; CHECK-DAG: lfs {{[0-9]+}}, 4(5)
; CHECK-DAG: lfs {{[0-9]+}}, 0(5)
; CHECK: blr
}

define i64* @_Z4testSt7complexIfE_idx(float %v0, float %v1, i64* %ref.tmp, float* %_M_value.realp.i.i, float* %_M_value.imagp.i.i) {
entry:
  %r = getelementptr i64, i64* %ref.tmp, i64 1
  %v2 = load i64, i64* %r, align 8
  %v3 = lshr i64 %v2, 32
  %v4 = trunc i64 %v3 to i32
  %v5 = bitcast i32 %v4 to float
  %v6 = trunc i64 %v2 to i32
  %v7 = bitcast i32 %v6 to float
  %mul_ad.i.i = fmul float %v5, %v1
  %mul_bc.i.i = fmul float %v7, %v0
  %mul_i.i.i = fadd float %mul_ad.i.i, %mul_bc.i.i
  %mul_ac.i.i = fmul float %v5, %v0
  %mul_bd.i.i = fmul float %v7, %v1
  %mul_r.i.i = fsub float %mul_ac.i.i, %mul_bd.i.i
  store float %mul_r.i.i, float* %_M_value.realp.i.i, align 4
  store float %mul_i.i.i, float* %_M_value.imagp.i.i, align 4
  ret i64* %r

; CHECK-LABEL: @_Z4testSt7complexIfE
; CHECK-NOT: ld {{[0-9]+}}, 8(5)
; CHECK-NOT: ldu {{[0-9]+}}, 8(5)
; CHECK-NOT: stw
; CHECK-NOT: rldicl
; CHECK-DAG: lfsu {{[0-9]+}}, 8(3)
; CHECK-DAG: lfs {{[0-9]+}}, 4(3)
; CHECK: blr
}

