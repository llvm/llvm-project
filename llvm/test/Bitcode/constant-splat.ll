; RUN: llvm-as -use-constant-int-for-fixed-length-splat \
; RUN:         -use-constant-fp-for-fixed-length-splat \
; RUN:         -use-constant-int-for-scalable-splat \
; RUN:         -use-constant-fp-for-scalable-splat \
; RUN:   < %s | llvm-dis -use-constant-int-for-fixed-length-splat \
; RUN:                   -use-constant-fp-for-fixed-length-splat \
; RUN:                   -use-constant-int-for-scalable-splat \
; RUN:                   -use-constant-fp-for-scalable-splat \
; RUN:   | FileCheck %s

; CHECK: @constant.splat.i1 = constant <1 x i1> splat (i1 true)
@constant.splat.i1 = constant <1 x i1> splat (i1 true)

; CHECK: @constant.splat.i32 = constant <5 x i32> splat (i32 7)
@constant.splat.i32 = constant <5 x i32> splat (i32 7)

; CHECK: @constant.splat.i128 = constant <7 x i128> splat (i128 85070591730234615870450834276742070272)
@constant.splat.i128 = constant <7 x i128> splat (i128 85070591730234615870450834276742070272)

; CHECK: @constant.splat.f16 = constant <2 x half> splat (half 0xHBC00)
@constant.splat.f16 = constant <2 x half> splat (half 0xHBC00)

; CHECK: @constant.splat.f32 = constant <4 x float> splat (float -2.000000e+00)
@constant.splat.f32 = constant <4 x float> splat (float -2.000000e+00)

; CHECK: @constant.splat.f64 = constant <6 x double> splat (double -3.000000e+00)
@constant.splat.f64 = constant <6 x double> splat (double -3.000000e+00)

; CHECK: @constant.splat.128 = constant <8 x fp128> splat (fp128 0xL00000000000000018000000000000000)
@constant.splat.128 = constant <8 x fp128> splat (fp128 0xL00000000000000018000000000000000)

; CHECK: @constant.splat.bf16 = constant <1 x bfloat> splat (bfloat 0xRC0A0)
@constant.splat.bf16 = constant <1 x bfloat> splat (bfloat 0xRC0A0)

; CHECK: @constant.splat.x86_fp80 = constant <3 x x86_fp80> splat (x86_fp80 0xK4000C8F5C28F5C28F800)
@constant.splat.x86_fp80 = constant <3 x x86_fp80> splat (x86_fp80 0xK4000C8F5C28F5C28F800)

; CHECK: @constant.splat.ppc_fp128 = constant <7 x ppc_fp128> splat (ppc_fp128 0xM80000000000000000000000000000000)
@constant.splat.ppc_fp128 = constant <7 x ppc_fp128> splat (ppc_fp128 0xM80000000000000000000000000000000)

define void @add_fixed_lenth_vector_splat_i32(<4 x i32> %a) {
; CHECK: %add = add <4 x i32> %a, splat (i32 137)
  %add = add <4 x i32> %a, splat (i32 137)
  ret void
}

define <4 x i32> @ret_fixed_lenth_vector_splat_i32() {
; CHECK: ret <4 x i32> splat (i32 56)
  ret <4 x i32> splat (i32 56)
}

define void @add_fixed_lenth_vector_splat_double(<vscale x 2 x double> %a) {
; CHECK: %add = fadd <vscale x 2 x double> %a, splat (double 5.700000e+00)
  %add = fadd <vscale x 2 x double> %a, splat (double 5.700000e+00)
  ret void
}

define <vscale x 4 x i32> @ret_scalable_vector_splat_i32() {
; CHECK: ret <vscale x 4 x i32> splat (i32 78)
  ret <vscale x 4 x i32> splat (i32 78)
}

define <4 x i32> @canonical_constant_vector() {
; CHECK: ret <4 x i32> splat (i32 7)
  ret <4 x i32> <i32 7, i32 7, i32 7, i32 7>
}

define <4 x i32> @canonical_fixed_lnegth_vector_zero() {
; CHECK: ret <4 x i32> zeroinitializer
  ret <4 x i32> zeroinitializer
}

define <vscale x 4 x i32> @canonical_scalable_lnegth_vector_zero() {
; CHECK: ret <vscale x 4 x i32> zeroinitializer
  ret <vscale x 4 x i32> zeroinitializer
}
