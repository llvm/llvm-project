; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

; NOTE: Tests the expansion of the "splat" shorthand method to create vector
; constants.  Future work will change how "splat" is expanded, ultimately
; leading to a point where "splat" is emitted as the disassembly.

@my_global = external global i32

; CHECK: @constant.splat.i1 = constant <1 x i1> splat (i1 true)
@constant.splat.i1 = constant <1 x i1> splat (i1 true)

; CHECK: @constant.splat.i32 = constant <5 x i32> splat (i32 7)
@constant.splat.i32 = constant <5 x i32> splat (i32 7)

; CHECK: @constant.splat.i128 = constant <2 x i128> splat (i128 85070591730234615870450834276742070272)
@constant.splat.i128 = constant <2 x i128> splat (i128 85070591730234615870450834276742070272)

; CHECK: @constant.splat.f16 = constant <4 x half> splat (half 0xHBC00)
@constant.splat.f16 = constant <4 x half> splat (half 0xHBC00)

; CHECK: @constant.splat.f32 = constant <5 x float> splat (float -2.000000e+00)
@constant.splat.f32 = constant <5 x float> splat (float -2.000000e+00)

; CHECK: @constant.splat.f64 = constant <3 x double> splat (double -3.000000e+00)
@constant.splat.f64 = constant <3 x double> splat (double -3.000000e+00)

; CHECK: @constant.splat.128 = constant <2 x fp128> splat (fp128 0xL00000000000000018000000000000000)
@constant.splat.128 = constant <2 x fp128> splat (fp128 0xL00000000000000018000000000000000)

; CHECK: @constant.splat.bf16 = constant <4 x bfloat> splat (bfloat 0xRC0A0)
@constant.splat.bf16 = constant <4 x bfloat> splat (bfloat 0xRC0A0)

; CHECK: @constant.splat.x86_fp80 = constant <3 x x86_fp80> splat (x86_fp80 0xK4000C8F5C28F5C28F800)
@constant.splat.x86_fp80 = constant <3 x x86_fp80> splat (x86_fp80 0xK4000C8F5C28F5C28F800)

; CHECK: @constant.splat.ppc_fp128 = constant <1 x ppc_fp128> splat (ppc_fp128 0xM80000000000000000000000000000000)
@constant.splat.ppc_fp128 = constant <1 x ppc_fp128> splat (ppc_fp128 0xM80000000000000000000000000000000)

; CHECK: @constant.splat.global.ptr = constant <4 x ptr> <ptr @my_global, ptr @my_global, ptr @my_global, ptr @my_global>
@constant.splat.global.ptr = constant <4 x ptr> splat (ptr @my_global)

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

define <vscale x 4 x ptr> @ret_scalable_vector_ptr() {
; CHECK: ret <vscale x 4 x ptr> shufflevector (<vscale x 4 x ptr> insertelement (<vscale x 4 x ptr> poison, ptr @my_global, i64 0), <vscale x 4 x ptr> poison, <vscale x 4 x i32> zeroinitializer)
  ret <vscale x 4 x ptr> splat (ptr @my_global)
}
