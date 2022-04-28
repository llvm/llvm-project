; Verify register types we generate in PTX.
; RUN: llc -O0 < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc -O0 < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: llc -O0 < %s -march=nvptx -mcpu=sm_20 | FileCheck %s -check-prefixes=NO8BIT
; RUN: llc -O0 < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s -check-prefixes=NO8BIT
; RUN: %if ptxas %{ llc -O0 < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc -O0 < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; CHECK-LABEL: .visible .func func()
; NO8BIT-LABEL: .visible .func func()
define void @func() {
entry:
  %s8 = alloca i8, align 1
  %u8 = alloca i8, align 1
  %s16 = alloca i16, align 2
  %u16 = alloca i16, align 2
; Both 8- and 16-bit integers are packed into 16-bit registers.
; CHECK-DAG: .reg .b16 %rs<
; We should not generate 8-bit registers.
; NO8BIT-NOT: .reg .{{[bsu]}}8
  %s32 = alloca i32, align 4
  %u32 = alloca i32, align 4
; CHECK-DAG: .reg .b32 %r<
  %s64 = alloca i64, align 8
  %u64 = alloca i64, align 8
; CHECK-DAG: .reg .b64 %rd<
  %f32 = alloca float, align 4
; CHECK-DAG: .reg .f32 %f<
  %f64 = alloca double, align 8
; CHECK-DAG: .reg .f64 %fd<

; Verify that we use correct register types.
  store i8 1, i8* %s8, align 1
; CHECK: mov.u16 [[R1:%rs[0-9]]], 1;
; CHECK-NEXT: st.u8 {{.*}}, [[R1]]
  store i8 2, i8* %u8, align 1
; CHECK: mov.u16 [[R2:%rs[0-9]]], 2;
; CHECK-NEXT: st.u8 {{.*}}, [[R2]]
  store i16 3, i16* %s16, align 2
; CHECK: mov.u16 [[R3:%rs[0-9]]], 3;
; CHECK-NEXT: st.u16 {{.*}}, [[R3]]
  store i16 4, i16* %u16, align 2
; CHECK: mov.u16 [[R4:%rs[0-9]]], 4;
; CHECK-NEXT: st.u16 {{.*}}, [[R4]]
  store i32 5, i32* %s32, align 4
; CHECK: mov.u32 [[R5:%r[0-9]]], 5;
; CHECK-NEXT: st.u32 {{.*}}, [[R5]]
  store i32 6, i32* %u32, align 4
; CHECK: mov.u32 [[R6:%r[0-9]]], 6;
; CHECK-NEXT: st.u32 {{.*}}, [[R6]]
  store i64 7, i64* %s64, align 8
; CHECK: mov.u64 [[R7:%rd[0-9]]], 7;
; CHECK-NEXT: st.u64 {{.*}}, [[R7]]
  store i64 8, i64* %u64, align 8
; CHECK: mov.u64 [[R8:%rd[0-9]]], 8;
; CHECK-NEXT: st.u64 {{.*}}, [[R8]]

; FP constants are stored via integer registers, but that's an
; implementation detail that's irrelevant here.
  store float 9.000000e+00, float* %f32, align 4
  store double 1.000000e+01, double* %f64, align 8
; Instead, we force a load into a register and then verify register type.
  %f32v = load volatile float, float* %f32, align 4
; CHECK: ld.volatile.f32         %f{{[0-9]+}}
  %f64v = load volatile double, double* %f64, align 8
; CHECK: ld.volatile.f64         %fd{{[0-9]+}}
  ret void
; CHECK: ret;
; NO8BIT: ret;
}

