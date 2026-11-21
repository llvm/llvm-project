; RUN: llc < %s -mcpu=sm_30 -mattr=+ptx42 | FileCheck %s --check-prefixes=CHECK,UNPACKED --implicit-check-not=mul.rn.f16
; RUN: llc < %s -mcpu=sm_53 -mattr=+ptx42 | FileCheck %s --check-prefixes=CHECK,UNPACKED --implicit-check-not=mul.rn.f16
; RUN: llc < %s -O0 -mcpu=sm_53 -mattr=+ptx42 | FileCheck %s --check-prefixes=CHECK,UNPACKED --implicit-check-not=mul.rn.f16
; RUN: llc < %s -mcpu=sm_100a -mattr=+ptx86 | FileCheck %s --check-prefixes=CHECK,PACKED --implicit-check-not=mul.rn.f16
; RUN: %if ptxas-sm_53 && ptxas-isa-4.2 %{ llc < %s -mcpu=sm_53 -mattr=+ptx42 | %ptxas-verify -arch=sm_53 %}
; RUN: %if ptxas-sm_100a && ptxas-isa-8.6 %{ llc < %s -mcpu=sm_100a -mattr=+ptx86 | %ptxas-verify -arch=sm_100a %}

target triple = "nvptx64-nvidia-cuda"

; The generic ldexp expansion must promote f16 to f32: expanding directly
; in f16 clamps the exponent to -20 and miscompiles smaller exponents.
define half @ldexp_f16(half %x, i32 %n) {
; CHECK-LABEL: ldexp_f16(
; CHECK:       cvt.f32.f16
; CHECK:       max.s32 {{.*}}, -330;
; CHECK:       mul.rn.f32 [[SCALED:%r[0-9]+]], {{.*}};
; CHECK-NEXT:  cvt.rn.f16.f32 [[RESULT:%rs[0-9]+]], [[SCALED]];
; CHECK-NEXT:  st.param{{(::func)?}}.b16 [func_retval0], [[RESULT]];
; CHECK-NEXT:  ret;
  %r = call half @llvm.ldexp.f16.i32(half %x, i32 %n)
  ret half %r
}

; For x = 1024, the result must be 2^-11, not 2^-10.
define half @ldexp_f16_neg21(half %x) {
; CHECK-LABEL: ldexp_f16_neg21(
; CHECK:       cvt.f32.f16 [[X:%r[0-9]+]], {{%rs[0-9]+}};
; CHECK-NEXT:  mul.rn.f32 [[SCALED:%r[0-9]+]], [[X]], 0f35000000;
; CHECK-NEXT:  cvt.rn.f16.f32 [[RESULT:%rs[0-9]+]], [[SCALED]];
; CHECK-NEXT:  st.param{{(::func)?}}.b16 [func_retval0], [[RESULT]];
; CHECK-NEXT:  ret;
  %r = call half @llvm.ldexp.f16.i32(half %x, i32 -21)
  ret half %r
}

; For x = 32768, the result must be 2^-14, not 2^-5.
define half @ldexp_f16_neg29(half %x) {
; CHECK-LABEL: ldexp_f16_neg29(
; CHECK:       cvt.f32.f16 [[X:%r[0-9]+]], {{%rs[0-9]+}};
; CHECK-NEXT:  mul.rn.f32 [[SCALED:%r[0-9]+]], [[X]], 0f31000000;
; CHECK-NEXT:  cvt.rn.f16.f32 [[RESULT:%rs[0-9]+]], [[SCALED]];
; CHECK-NEXT:  st.param{{(::func)?}}.b16 [func_retval0], [[RESULT]];
; CHECK-NEXT:  ret;
  %r = call half @llvm.ldexp.f16.i32(half %x, i32 -29)
  ret half %r
}

; The largest finite input, 65504, rounds to the smallest subnormal, 2^-24.
define half @ldexp_f16_neg40(half %x) {
; CHECK-LABEL: ldexp_f16_neg40(
; CHECK:       cvt.f32.f16 [[X:%r[0-9]+]], {{%rs[0-9]+}};
; CHECK-NEXT:  mul.rn.f32 [[SCALED:%r[0-9]+]], [[X]], 0f2B800000;
; CHECK-NEXT:  cvt.rn.f16.f32 [[RESULT:%rs[0-9]+]], [[SCALED]];
; CHECK-NEXT:  st.param{{(::func)?}}.b16 [func_retval0], [[RESULT]];
; CHECK-NEXT:  ret;
  %r = call half @llvm.ldexp.f16.i32(half %x, i32 -40)
  ret half %r
}

; Even the largest finite input must round to zero at this exponent.
define half @ldexp_f16_neg41(half %x) {
; CHECK-LABEL: ldexp_f16_neg41(
; CHECK:       cvt.f32.f16 [[X:%r[0-9]+]], {{%rs[0-9]+}};
; CHECK-NEXT:  mul.rn.f32 [[SCALED:%r[0-9]+]], [[X]], 0f2B000000;
; CHECK-NEXT:  cvt.rn.f16.f32 [[RESULT:%rs[0-9]+]], [[SCALED]];
; CHECK-NEXT:  st.param{{(::func)?}}.b16 [func_retval0], [[RESULT]];
; CHECK-NEXT:  ret;
  %r = call half @llvm.ldexp.f16.i32(half %x, i32 -41)
  ret half %r
}

; Every finite f16 input must underflow to zero for this exponent.
define half @ldexp_f16_neg100(half %x) {
; CHECK-LABEL: ldexp_f16_neg100(
; CHECK:       cvt.f32.f16 [[X:%r[0-9]+]], {{%rs[0-9]+}};
; CHECK-NEXT:  mul.rn.f32 [[SCALED:%r[0-9]+]], [[X]], 0f0D800000;
; CHECK-NEXT:  cvt.rn.f16.f32 [[RESULT:%rs[0-9]+]], [[SCALED]];
; CHECK-NEXT:  st.param{{(::func)?}}.b16 [func_retval0], [[RESULT]];
; CHECK-NEXT:  ret;
  %r = call half @llvm.ldexp.f16.i32(half %x, i32 -100)
  ret half %r
}

; Scaling the smallest subnormal by 2^24 must produce 1.0.
define half @ldexp_f16_pos24(half %x) {
; CHECK-LABEL: ldexp_f16_pos24(
; CHECK:       cvt.f32.f16 [[X:%r[0-9]+]], {{%rs[0-9]+}};
; CHECK-NEXT:  mul.rn.f32 [[SCALED:%r[0-9]+]], [[X]], 0f4B800000;
; CHECK-NEXT:  cvt.rn.f16.f32 [[RESULT:%rs[0-9]+]], [[SCALED]];
; CHECK-NEXT:  st.param{{(::func)?}}.b16 [func_retval0], [[RESULT]];
; CHECK-NEXT:  ret;
  %r = call half @llvm.ldexp.f16.i32(half %x, i32 24)
  ret half %r
}

define <2 x half> @ldexp_v2f16(<2 x half> %x, <2 x i32> %n) {
; CHECK-LABEL: ldexp_v2f16(
; CHECK:         cvt.f32.f16
; CHECK:         max.s32 {{.*}}, -330;
; CHECK:         mul.rn.f32 [[SCALED0:%r[0-9]+]], {{.*}};
; UNPACKED-NEXT: cvt.rn.f16.f32 [[RESULT0:%rs[0-9]+]], [[SCALED0]];
; CHECK:         cvt.f32.f16
; CHECK:         max.s32 {{.*}}, -330;
; CHECK:         mul.rn.f32 [[SCALED1:%r[0-9]+]], {{.*}};
; UNPACKED-NEXT: cvt.rn.f16.f32 [[RESULT1:%rs[0-9]+]], [[SCALED1]];
; UNPACKED-NEXT: mov.b32 [[RESULT:%r[0-9]+]], {[[RESULT1]], [[RESULT0]]};
; PACKED-NEXT:   cvt.rn.f16x2.f32 [[RESULT:%r[0-9]+]], [[SCALED1]], [[SCALED0]];
; CHECK-NEXT:    st.param{{(::func)?}}.b32 [func_retval0], [[RESULT]];
; CHECK-NEXT:    ret;
  %r = call <2 x half> @llvm.ldexp.v2f16.v2i32(<2 x half> %x, <2 x i32> %n)
  ret <2 x half> %r
}

declare half @llvm.ldexp.f16.i32(half, i32)
declare <2 x half> @llvm.ldexp.v2f16.v2i32(<2 x half>, <2 x i32>)
