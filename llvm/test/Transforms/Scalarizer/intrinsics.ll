; RUN: opt -S -passes='function(scalarizer)' %s | FileCheck %s

; Unary fp
declare <2 x float> @llvm.sqrt.v2f32(<2 x float>)

; Binary fp
declare <2 x float> @llvm.minnum.v2f32(<2 x float>, <2 x float>)
declare <2 x float> @llvm.minimum.v2f32(<2 x float>, <2 x float>)
declare <2 x float> @llvm.maximum.v2f32(<2 x float>, <2 x float>)

; Ternary fp
declare <2 x float> @llvm.fma.v2f32(<2 x float>, <2 x float>, <2 x float>)

; Unary int
declare <2 x i32> @llvm.bswap.v2i32(<2 x i32>)

; Unary int plus constant scalar operand
declare <2 x i32> @llvm.ctlz.v2i32(<2 x i32>, i1)

; Unary fp plus any scalar operand
declare <2 x float> @llvm.powi.v2f32.i32(<2 x float>, i32)

; Binary int plus constant scalar operand
declare <2 x i32> @llvm.smul.fix.sat.v2i32(<2 x i32>, <2 x i32>, i32)
declare <2 x i32> @llvm.umul.fix.sat.v2i32(<2 x i32>, <2 x i32>, i32)

declare <2 x i32> @llvm.fptosi.sat.v2i32.v2f32(<2 x float>)
declare <2 x i32> @llvm.fptoui.sat.v2i32.v2f32(<2 x float>)


; CHECK-LABEL: @scalarize_sqrt_v2f32(
; CHECK: %sqrt.i0 = call float @llvm.sqrt.f32(float %x.i0)
; CHECK: %sqrt.i1 = call float @llvm.sqrt.f32(float %x.i1)
; CHECK: %sqrt.upto0 = insertelement <2 x float> poison, float %sqrt.i0, i32 0
; CHECK: %sqrt = insertelement <2 x float> %sqrt.upto0, float %sqrt.i1, i32 1
; CHECK: ret <2 x float> %sqrt
define <2 x float> @scalarize_sqrt_v2f32(<2 x float> %x) #0 {
  %sqrt = call <2 x float> @llvm.sqrt.v2f32(<2 x float> %x)
  ret <2 x float> %sqrt
}

; CHECK-LABEL: @scalarize_minnum_v2f32(
; CHECK: %minnum.i0 = call float @llvm.minnum.f32(float %x.i0, float %y.i0)
; CHECK: %minnum.i1 = call float @llvm.minnum.f32(float %x.i1, float %y.i1)
; CHECK: %minnum.upto0 = insertelement <2 x float> poison, float %minnum.i0, i32 0
; CHECK: %minnum = insertelement <2 x float> %minnum.upto0, float %minnum.i1, i32 1
; CHECK: ret <2 x float> %minnum
define <2 x float> @scalarize_minnum_v2f32(<2 x float> %x, <2 x float> %y) #0 {
  %minnum = call <2 x float> @llvm.minnum.v2f32(<2 x float> %x, <2 x float> %y)
  ret <2 x float> %minnum
}

; CHECK-LABEL: @scalarize_minimum_v2f32(
; CHECK: %minimum.i0 = call float @llvm.minimum.f32(float %x.i0, float %y.i0)
; CHECK: %minimum.i1 = call float @llvm.minimum.f32(float %x.i1, float %y.i1)
; CHECK: %minimum.upto0 = insertelement <2 x float> poison, float %minimum.i0, i32 0
; CHECK: %minimum = insertelement <2 x float> %minimum.upto0, float %minimum.i1, i32 1
; CHECK: ret <2 x float> %minimum
define <2 x float> @scalarize_minimum_v2f32(<2 x float> %x, <2 x float> %y) #0 {
  %minimum = call <2 x float> @llvm.minimum.v2f32(<2 x float> %x, <2 x float> %y)
  ret <2 x float> %minimum
}

; CHECK-LABEL: @scalarize_maximum_v2f32(
; CHECK: %maximum.i0 = call float @llvm.maximum.f32(float %x.i0, float %y.i0)
; CHECK: %maximum.i1 = call float @llvm.maximum.f32(float %x.i1, float %y.i1)
; CHECK: %maximum.upto0 = insertelement <2 x float> poison, float %maximum.i0, i32 0
; CHECK: %maximum = insertelement <2 x float> %maximum.upto0, float %maximum.i1, i32 1
; CHECK: ret <2 x float> %maximum
define <2 x float> @scalarize_maximum_v2f32(<2 x float> %x, <2 x float> %y) #0 {
  %maximum = call <2 x float> @llvm.maximum.v2f32(<2 x float> %x, <2 x float> %y)
  ret <2 x float> %maximum
}

; CHECK-LABEL: @scalarize_fma_v2f32(
; CHECK: %fma.i0 = call float @llvm.fma.f32(float %x.i0, float %y.i0, float %z.i0)
; CHECK: %fma.i1 = call float @llvm.fma.f32(float %x.i1, float %y.i1, float %z.i1)
; CHECK: %fma.upto0 = insertelement <2 x float> poison, float %fma.i0, i32 0
; CHECK: %fma = insertelement <2 x float> %fma.upto0, float %fma.i1, i32 1
; CHECK: ret <2 x float> %fma
define <2 x float> @scalarize_fma_v2f32(<2 x float> %x, <2 x float> %y, <2 x float> %z) #0 {
  %fma = call <2 x float> @llvm.fma.v2f32(<2 x float> %x, <2 x float> %y, <2 x float> %z)
  ret <2 x float> %fma
}

; CHECK-LABEL: @scalarize_bswap_v2i32(
; CHECK: %bswap.i0 = call i32 @llvm.bswap.i32(i32 %x.i0)
; CHECK: %bswap.i1 = call i32 @llvm.bswap.i32(i32 %x.i1)
; CHECK: %bswap.upto0 = insertelement <2 x i32> poison, i32 %bswap.i0, i32 0
; CHECK: %bswap = insertelement <2 x i32> %bswap.upto0, i32 %bswap.i1, i32 1
; CHECK: ret <2 x i32> %bswap
define <2 x i32> @scalarize_bswap_v2i32(<2 x i32> %x) #0 {
  %bswap = call <2 x i32> @llvm.bswap.v2i32(<2 x i32> %x)
  ret <2 x i32> %bswap
}

; CHECK-LABEL: @scalarize_ctlz_v2i32(
; CHECK: %ctlz.i0 = call i32 @llvm.ctlz.i32(i32 %x.i0, i1 true)
; CHECK: %ctlz.i1 = call i32 @llvm.ctlz.i32(i32 %x.i1, i1 true)
; CHECK: %ctlz.upto0 = insertelement <2 x i32> poison, i32 %ctlz.i0, i32 0
; CHECK: %ctlz = insertelement <2 x i32> %ctlz.upto0, i32 %ctlz.i1, i32 1
; CHECK: ret <2 x i32> %ctlz
define <2 x i32> @scalarize_ctlz_v2i32(<2 x i32> %x) #0 {
  %ctlz = call <2 x i32> @llvm.ctlz.v2i32(<2 x i32> %x, i1 true)
  ret <2 x i32> %ctlz
}

; CHECK-LABEL: @scalarize_powi_v2f32(
; CHECK: %powi.i0 = call float @llvm.powi.f32.i32(float %x.i0, i32 %y)
; CHECK: %powi.i1 = call float @llvm.powi.f32.i32(float %x.i1, i32 %y)
; CHECK: %powi.upto0 = insertelement <2 x float> poison, float %powi.i0, i32 0
; CHECK: %powi = insertelement <2 x float> %powi.upto0, float %powi.i1, i32 1
; CHECK: ret <2 x float> %powi
define <2 x float> @scalarize_powi_v2f32(<2 x float> %x, i32 %y) #0 {
  %powi = call <2 x float> @llvm.powi.v2f32.i32(<2 x float> %x, i32 %y)
  ret <2 x float> %powi
}

; CHECK-LABEL: @scalarize_smul_fix_sat_v2i32(
; CHECK: %smulfixsat.i0 = call i32 @llvm.smul.fix.sat.i32(i32 %x.i0, i32 5, i32 31)
; CHECK: %smulfixsat.i1 = call i32 @llvm.smul.fix.sat.i32(i32 %x.i1, i32 19, i32 31)
; CHECK: %smulfixsat.upto0 = insertelement <2 x i32> poison, i32 %smulfixsat.i0, i32 0
; CHECK: %smulfixsat = insertelement <2 x i32> %smulfixsat.upto0, i32 %smulfixsat.i1, i32 1
; CHECK: ret <2 x i32> %smulfixsat
define <2 x i32> @scalarize_smul_fix_sat_v2i32(<2 x i32> %x) #0 {
  %smulfixsat = call <2 x i32> @llvm.smul.fix.sat.v2i32(<2 x i32> %x, <2 x i32> <i32 5, i32 19>, i32 31)
  ret <2 x i32> %smulfixsat
}

; CHECK-LABEL: @scalarize_umul_fix_sat_v2i32(
; CHECK: %umulfixsat.i0 = call i32 @llvm.umul.fix.sat.i32(i32 %x.i0, i32 5, i32 31)
; CHECK: %umulfixsat.i1 = call i32 @llvm.umul.fix.sat.i32(i32 %x.i1, i32 19, i32 31)
; CHECK: %umulfixsat.upto0 = insertelement <2 x i32> poison, i32 %umulfixsat.i0, i32 0
; CHECK: %umulfixsat = insertelement <2 x i32> %umulfixsat.upto0, i32 %umulfixsat.i1, i32 1
; CHECK: ret <2 x i32> %umulfixsat
define <2 x i32> @scalarize_umul_fix_sat_v2i32(<2 x i32> %x) #0 {
  %umulfixsat = call <2 x i32> @llvm.umul.fix.sat.v2i32(<2 x i32> %x, <2 x i32> <i32 5, i32 19>, i32 31)
  ret <2 x i32> %umulfixsat
}

; CHECK-LABEL: @scalarize_fptosi_sat(
; CHECK: %sat.i0 = call i32 @llvm.fptosi.sat.i32.f32(float %x.i0)
; CHECK: %sat.i1 = call i32 @llvm.fptosi.sat.i32.f32(float %x.i1)
; CHECK: %sat.upto0 = insertelement <2 x i32> poison, i32 %sat.i0, i32 0
; CHECK: %sat = insertelement <2 x i32> %sat.upto0, i32 %sat.i1, i32 1
; CHECK: ret <2 x i32> %sat
define <2 x i32> @scalarize_fptosi_sat(<2 x float> %x) #0 {
  %sat = call <2 x i32> @llvm.fptosi.sat.v2i32.v2f32(<2 x float> %x)
  ret <2 x i32> %sat
}

; CHECK-LABEL: @scalarize_fptoui_sat(
; CHECK: %sat.i0 = call i32 @llvm.fptoui.sat.i32.f32(float %x.i0)
; CHECK: %sat.i1 = call i32 @llvm.fptoui.sat.i32.f32(float %x.i1)
; CHECK: %sat.upto0 = insertelement <2 x i32> poison, i32 %sat.i0, i32 0
; CHECK: %sat = insertelement <2 x i32> %sat.upto0, i32 %sat.i1, i32 1
; CHECK: ret <2 x i32> %sat
define <2 x i32> @scalarize_fptoui_sat(<2 x float> %x) #0 {
  %sat = call <2 x i32> @llvm.fptoui.sat.v2i32.v2f32(<2 x float> %x)
  ret <2 x i32> %sat
}
