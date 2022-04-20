; RUN: llc < %s -mtriple=nvptx-unknown-unknown | FileCheck %s
;
; Check that parameters of a __global__ (kernel) function do not get increased
; alignment, and no additional vectorization is performed on loads/stores with
; that parameters.
;
; Test IR is a minimized version of IR generated with the following command
; from the source code below:
; $ clang++ -O3 --cuda-gpu-arch=sm_35 -S -emit-llvm src.cu
;
; ----------------------------------------------------------------------------
; #include <stdint.h>
;
; struct St4x1 { uint32_t field[1]; };
; struct St4x2 { uint32_t field[2]; };
; struct St4x3 { uint32_t field[3]; };
; struct St4x4 { uint32_t field[4]; };
; struct St4x5 { uint32_t field[5]; };
; struct St4x6 { uint32_t field[6]; };
; struct St4x7 { uint32_t field[7]; };
; struct St4x8 { uint32_t field[8]; };
; struct St8x1 { uint64_t field[1]; };
; struct St8x2 { uint64_t field[2]; };
; struct St8x3 { uint64_t field[3]; };
; struct St8x4 { uint64_t field[4]; };
;
; #define DECLARE_FUNCTION(StName)                                    \
; static __global__  __attribute__((noinline))                        \
; void foo_##StName(struct StName in, struct StName* ret) {           \
;   const unsigned size = sizeof(ret->field) / sizeof(*ret->field);   \
;   for (unsigned i = 0; i != size; ++i)                              \
;     ret->field[i] = in.field[i];                                    \
; }                                                                   \
;
; DECLARE_FUNCTION(St4x1)
; DECLARE_FUNCTION(St4x2)
; DECLARE_FUNCTION(St4x3)
; DECLARE_FUNCTION(St4x4)
; DECLARE_FUNCTION(St4x5)
; DECLARE_FUNCTION(St4x6)
; DECLARE_FUNCTION(St4x7)
; DECLARE_FUNCTION(St4x8)
; DECLARE_FUNCTION(St8x1)
; DECLARE_FUNCTION(St8x2)
; DECLARE_FUNCTION(St8x3)
; DECLARE_FUNCTION(St8x4)
; ----------------------------------------------------------------------------

%struct.St4x1 = type { [1 x i32] }
%struct.St4x2 = type { [2 x i32] }
%struct.St4x3 = type { [3 x i32] }
%struct.St4x4 = type { [4 x i32] }
%struct.St4x5 = type { [5 x i32] }
%struct.St4x6 = type { [6 x i32] }
%struct.St4x7 = type { [7 x i32] }
%struct.St4x8 = type { [8 x i32] }
%struct.St8x1 = type { [1 x i64] }
%struct.St8x2 = type { [2 x i64] }
%struct.St8x3 = type { [3 x i64] }
%struct.St8x4 = type { [4 x i64] }

define dso_local void @foo_St4x1(ptr nocapture noundef readonly byval(%struct.St4x1) align 4 %in, ptr nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func foo_St4x1(
  ; CHECK:               .param .align 4 .b8 foo_St4x1_param_0[4],
  ; CHECK:               .param .b32 foo_St4x1_param_1
  ; CHECK:       )
  ; CHECK:       ld.param.u32 [[R1:%r[0-9]+]], [foo_St4x1_param_1];
  ; CHECK:       ld.param.u32 [[R2:%r[0-9]+]], [foo_St4x1_param_0];
  ; CHECK:       st.u32  [[[R1]]], [[R2]];
  ; CHECK:       ret;
  %1 = load i32, ptr %in, align 4
  store i32 %1, ptr %ret, align 4
  ret void
}

define dso_local void @foo_St4x2(ptr nocapture noundef readonly byval(%struct.St4x2) align 4 %in, ptr nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func foo_St4x2(
  ; CHECK:               .param .align 4 .b8 foo_St4x2_param_0[8],
  ; CHECK:               .param .b32 foo_St4x2_param_1
  ; CHECK:       )
  ; CHECK:       ld.param.u32 [[R1:%r[0-9]+]], [foo_St4x2_param_1];
  ; CHECK:       ld.param.u32 [[R2:%r[0-9]+]], [foo_St4x2_param_0];
  ; CHECK:       st.u32  [[[R1]]], [[R2]];
  ; CHECK:       ld.param.u32 [[R3:%r[0-9]+]], [foo_St4x2_param_0+4];
  ; CHECK:       st.u32  [[[R1]]+4], [[R3]];
  ; CHECK:       ret;
  %1 = load i32, ptr %in, align 4
  store i32 %1, ptr %ret, align 4
  %arrayidx.1 = getelementptr inbounds [2 x i32], ptr %in, i64 0, i64 1
  %2 = load i32, ptr %arrayidx.1, align 4
  %arrayidx3.1 = getelementptr inbounds [2 x i32], ptr %ret, i64 0, i64 1
  store i32 %2, ptr %arrayidx3.1, align 4
  ret void
}

define dso_local void @foo_St4x3(ptr nocapture noundef readonly byval(%struct.St4x3) align 4 %in, ptr nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func foo_St4x3(
  ; CHECK:               .param .align 4 .b8 foo_St4x3_param_0[12],
  ; CHECK:               .param .b32 foo_St4x3_param_1
  ; CHECK:       )
  ; CHECK:       ld.param.u32 [[R1:%r[0-9]+]], [foo_St4x3_param_1];
  ; CHECK:       ld.param.u32 [[R2:%r[0-9]+]], [foo_St4x3_param_0];
  ; CHECK:       st.u32  [[[R1]]], [[R2]];
  ; CHECK:       ld.param.u32 [[R3:%r[0-9]+]], [foo_St4x3_param_0+4];
  ; CHECK:       st.u32  [[[R1]]+4], [[R3]];
  ; CHECK:       ld.param.u32 [[R4:%r[0-9]+]], [foo_St4x3_param_0+8];
  ; CHECK:       st.u32  [[[R1]]+8], [[R4]];
  ; CHECK:       ret;
  %1 = load i32, ptr %in, align 4
  store i32 %1, ptr %ret, align 4
  %arrayidx.1 = getelementptr inbounds [3 x i32], ptr %in, i64 0, i64 1
  %2 = load i32, ptr %arrayidx.1, align 4
  %arrayidx3.1 = getelementptr inbounds [3 x i32], ptr %ret, i64 0, i64 1
  store i32 %2, ptr %arrayidx3.1, align 4
  %arrayidx.2 = getelementptr inbounds [3 x i32], ptr %in, i64 0, i64 2
  %3 = load i32, ptr %arrayidx.2, align 4
  %arrayidx3.2 = getelementptr inbounds [3 x i32], ptr %ret, i64 0, i64 2
  store i32 %3, ptr %arrayidx3.2, align 4
  ret void
}

define dso_local void @foo_St4x4(ptr nocapture noundef readonly byval(%struct.St4x4) align 4 %in, ptr nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func foo_St4x4(
  ; CHECK:               .param .align 4 .b8 foo_St4x4_param_0[16],
  ; CHECK:               .param .b32 foo_St4x4_param_1
  ; CHECK:       )
  ; CHECK:       ld.param.u32 [[R1:%r[0-9]+]], [foo_St4x4_param_1];
  ; CHECK:       ld.param.u32 [[R2:%r[0-9]+]], [foo_St4x4_param_0];
  ; CHECK:       st.u32  [[[R1]]], [[R2]];
  ; CHECK:       ld.param.u32 [[R3:%r[0-9]+]], [foo_St4x4_param_0+4];
  ; CHECK:       st.u32  [[[R1]]+4], [[R3]];
  ; CHECK:       ld.param.u32 [[R4:%r[0-9]+]], [foo_St4x4_param_0+8];
  ; CHECK:       st.u32  [[[R1]]+8], [[R4]];
  ; CHECK:       ld.param.u32 [[R5:%r[0-9]+]], [foo_St4x4_param_0+12];
  ; CHECK:       st.u32  [[[R1]]+12], [[R5]];
  ; CHECK:       ret;
  %1 = load i32, ptr %in, align 4
  store i32 %1, ptr %ret, align 4
  %arrayidx.1 = getelementptr inbounds [4 x i32], ptr %in, i64 0, i64 1
  %2 = load i32, ptr %arrayidx.1, align 4
  %arrayidx3.1 = getelementptr inbounds [4 x i32], ptr %ret, i64 0, i64 1
  store i32 %2, ptr %arrayidx3.1, align 4
  %arrayidx.2 = getelementptr inbounds [4 x i32], ptr %in, i64 0, i64 2
  %3 = load i32, ptr %arrayidx.2, align 4
  %arrayidx3.2 = getelementptr inbounds [4 x i32], ptr %ret, i64 0, i64 2
  store i32 %3, ptr %arrayidx3.2, align 4
  %arrayidx.3 = getelementptr inbounds [4 x i32], ptr %in, i64 0, i64 3
  %4 = load i32, ptr %arrayidx.3, align 4
  %arrayidx3.3 = getelementptr inbounds [4 x i32], ptr %ret, i64 0, i64 3
  store i32 %4, ptr %arrayidx3.3, align 4
  ret void
}

define dso_local void @foo_St4x5(ptr nocapture noundef readonly byval(%struct.St4x5) align 4 %in, ptr nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func foo_St4x5(
  ; CHECK:               .param .align 4 .b8 foo_St4x5_param_0[20],
  ; CHECK:               .param .b32 foo_St4x5_param_1
  ; CHECK:       )
  ; CHECK:       ld.param.u32 [[R1:%r[0-9]+]], [foo_St4x5_param_1];
  ; CHECK:       ld.param.u32 [[R2:%r[0-9]+]], [foo_St4x5_param_0];
  ; CHECK:       st.u32  [[[R1]]], [[R2]];
  ; CHECK:       ld.param.u32 [[R3:%r[0-9]+]], [foo_St4x5_param_0+4];
  ; CHECK:       st.u32  [[[R1]]+4], [[R3]];
  ; CHECK:       ld.param.u32 [[R4:%r[0-9]+]], [foo_St4x5_param_0+8];
  ; CHECK:       st.u32  [[[R1]]+8], [[R4]];
  ; CHECK:       ld.param.u32 [[R5:%r[0-9]+]], [foo_St4x5_param_0+12];
  ; CHECK:       st.u32  [[[R1]]+12], [[R5]];
  ; CHECK:       ld.param.u32 [[R6:%r[0-9]+]], [foo_St4x5_param_0+16];
  ; CHECK:       st.u32  [[[R1]]+16], [[R6]];
  ; CHECK:       ret;
  %1 = load i32, ptr %in, align 4
  store i32 %1, ptr %ret, align 4
  %arrayidx.1 = getelementptr inbounds [5 x i32], ptr %in, i64 0, i64 1
  %2 = load i32, ptr %arrayidx.1, align 4
  %arrayidx3.1 = getelementptr inbounds [5 x i32], ptr %ret, i64 0, i64 1
  store i32 %2, ptr %arrayidx3.1, align 4
  %arrayidx.2 = getelementptr inbounds [5 x i32], ptr %in, i64 0, i64 2
  %3 = load i32, ptr %arrayidx.2, align 4
  %arrayidx3.2 = getelementptr inbounds [5 x i32], ptr %ret, i64 0, i64 2
  store i32 %3, ptr %arrayidx3.2, align 4
  %arrayidx.3 = getelementptr inbounds [5 x i32], ptr %in, i64 0, i64 3
  %4 = load i32, ptr %arrayidx.3, align 4
  %arrayidx3.3 = getelementptr inbounds [5 x i32], ptr %ret, i64 0, i64 3
  store i32 %4, ptr %arrayidx3.3, align 4
  %arrayidx.4 = getelementptr inbounds [5 x i32], ptr %in, i64 0, i64 4
  %5 = load i32, ptr %arrayidx.4, align 4
  %arrayidx3.4 = getelementptr inbounds [5 x i32], ptr %ret, i64 0, i64 4
  store i32 %5, ptr %arrayidx3.4, align 4
  ret void
}

define dso_local void @foo_St4x6(ptr nocapture noundef readonly byval(%struct.St4x6) align 4 %in, ptr nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func foo_St4x6(
  ; CHECK:               .param .align 4 .b8 foo_St4x6_param_0[24],
  ; CHECK:               .param .b32 foo_St4x6_param_1
  ; CHECK:       )
  ; CHECK:       ld.param.u32 [[R1:%r[0-9]+]], [foo_St4x6_param_1];
  ; CHECK:       ld.param.u32 [[R2:%r[0-9]+]], [foo_St4x6_param_0];
  ; CHECK:       st.u32  [[[R1]]], [[R2]];
  ; CHECK:       ld.param.u32 [[R3:%r[0-9]+]], [foo_St4x6_param_0+4];
  ; CHECK:       st.u32  [[[R1]]+4], [[R3]];
  ; CHECK:       ld.param.u32 [[R4:%r[0-9]+]], [foo_St4x6_param_0+8];
  ; CHECK:       st.u32  [[[R1]]+8], [[R4]];
  ; CHECK:       ld.param.u32 [[R5:%r[0-9]+]], [foo_St4x6_param_0+12];
  ; CHECK:       st.u32  [[[R1]]+12], [[R5]];
  ; CHECK:       ld.param.u32 [[R6:%r[0-9]+]], [foo_St4x6_param_0+16];
  ; CHECK:       st.u32  [[[R1]]+16], [[R6]];
  ; CHECK:       ld.param.u32 [[R7:%r[0-9]+]], [foo_St4x6_param_0+20];
  ; CHECK:       st.u32  [[[R1]]+20], [[R7]];
  ; CHECK:       ret;
  %1 = load i32, ptr %in, align 4
  store i32 %1, ptr %ret, align 4
  %arrayidx.1 = getelementptr inbounds [6 x i32], ptr %in, i64 0, i64 1
  %2 = load i32, ptr %arrayidx.1, align 4
  %arrayidx3.1 = getelementptr inbounds [6 x i32], ptr %ret, i64 0, i64 1
  store i32 %2, ptr %arrayidx3.1, align 4
  %arrayidx.2 = getelementptr inbounds [6 x i32], ptr %in, i64 0, i64 2
  %3 = load i32, ptr %arrayidx.2, align 4
  %arrayidx3.2 = getelementptr inbounds [6 x i32], ptr %ret, i64 0, i64 2
  store i32 %3, ptr %arrayidx3.2, align 4
  %arrayidx.3 = getelementptr inbounds [6 x i32], ptr %in, i64 0, i64 3
  %4 = load i32, ptr %arrayidx.3, align 4
  %arrayidx3.3 = getelementptr inbounds [6 x i32], ptr %ret, i64 0, i64 3
  store i32 %4, ptr %arrayidx3.3, align 4
  %arrayidx.4 = getelementptr inbounds [6 x i32], ptr %in, i64 0, i64 4
  %5 = load i32, ptr %arrayidx.4, align 4
  %arrayidx3.4 = getelementptr inbounds [6 x i32], ptr %ret, i64 0, i64 4
  store i32 %5, ptr %arrayidx3.4, align 4
  %arrayidx.5 = getelementptr inbounds [6 x i32], ptr %in, i64 0, i64 5
  %6 = load i32, ptr %arrayidx.5, align 4
  %arrayidx3.5 = getelementptr inbounds [6 x i32], ptr %ret, i64 0, i64 5
  store i32 %6, ptr %arrayidx3.5, align 4
  ret void
}

define dso_local void @foo_St4x7(ptr nocapture noundef readonly byval(%struct.St4x7) align 4 %in, ptr nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func foo_St4x7(
  ; CHECK:               .param .align 4 .b8 foo_St4x7_param_0[28],
  ; CHECK:               .param .b32 foo_St4x7_param_1
  ; CHECK:       )
  ; CHECK:       ld.param.u32 [[R1:%r[0-9]+]], [foo_St4x7_param_1];
  ; CHECK:       ld.param.u32 [[R2:%r[0-9]+]], [foo_St4x7_param_0];
  ; CHECK:       st.u32  [[[R1]]], [[R2]];
  ; CHECK:       ld.param.u32 [[R3:%r[0-9]+]], [foo_St4x7_param_0+4];
  ; CHECK:       st.u32  [[[R1]]+4], [[R3]];
  ; CHECK:       ld.param.u32 [[R4:%r[0-9]+]], [foo_St4x7_param_0+8];
  ; CHECK:       st.u32  [[[R1]]+8], [[R4]];
  ; CHECK:       ld.param.u32 [[R5:%r[0-9]+]], [foo_St4x7_param_0+12];
  ; CHECK:       st.u32  [[[R1]]+12], [[R5]];
  ; CHECK:       ld.param.u32 [[R6:%r[0-9]+]], [foo_St4x7_param_0+16];
  ; CHECK:       st.u32  [[[R1]]+16], [[R6]];
  ; CHECK:       ld.param.u32 [[R7:%r[0-9]+]], [foo_St4x7_param_0+20];
  ; CHECK:       st.u32  [[[R1]]+20], [[R7]];
  ; CHECK:       ld.param.u32 [[R8:%r[0-9]+]], [foo_St4x7_param_0+24];
  ; CHECK:       st.u32  [[[R1]]+24], [[R8]];
  ; CHECK:       ret;
  %1 = load i32, ptr %in, align 4
  store i32 %1, ptr %ret, align 4
  %arrayidx.1 = getelementptr inbounds [7 x i32], ptr %in, i64 0, i64 1
  %2 = load i32, ptr %arrayidx.1, align 4
  %arrayidx3.1 = getelementptr inbounds [7 x i32], ptr %ret, i64 0, i64 1
  store i32 %2, ptr %arrayidx3.1, align 4
  %arrayidx.2 = getelementptr inbounds [7 x i32], ptr %in, i64 0, i64 2
  %3 = load i32, ptr %arrayidx.2, align 4
  %arrayidx3.2 = getelementptr inbounds [7 x i32], ptr %ret, i64 0, i64 2
  store i32 %3, ptr %arrayidx3.2, align 4
  %arrayidx.3 = getelementptr inbounds [7 x i32], ptr %in, i64 0, i64 3
  %4 = load i32, ptr %arrayidx.3, align 4
  %arrayidx3.3 = getelementptr inbounds [7 x i32], ptr %ret, i64 0, i64 3
  store i32 %4, ptr %arrayidx3.3, align 4
  %arrayidx.4 = getelementptr inbounds [7 x i32], ptr %in, i64 0, i64 4
  %5 = load i32, ptr %arrayidx.4, align 4
  %arrayidx3.4 = getelementptr inbounds [7 x i32], ptr %ret, i64 0, i64 4
  store i32 %5, ptr %arrayidx3.4, align 4
  %arrayidx.5 = getelementptr inbounds [7 x i32], ptr %in, i64 0, i64 5
  %6 = load i32, ptr %arrayidx.5, align 4
  %arrayidx3.5 = getelementptr inbounds [7 x i32], ptr %ret, i64 0, i64 5
  store i32 %6, ptr %arrayidx3.5, align 4
  %arrayidx.6 = getelementptr inbounds [7 x i32], ptr %in, i64 0, i64 6
  %7 = load i32, ptr %arrayidx.6, align 4
  %arrayidx3.6 = getelementptr inbounds [7 x i32], ptr %ret, i64 0, i64 6
  store i32 %7, ptr %arrayidx3.6, align 4
  ret void
}

define dso_local void @foo_St4x8(ptr nocapture noundef readonly byval(%struct.St4x8) align 4 %in, ptr nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func foo_St4x8(
  ; CHECK:               .param .align 4 .b8 foo_St4x8_param_0[32],
  ; CHECK:               .param .b32 foo_St4x8_param_1
  ; CHECK:       )
  ; CHECK:       ld.param.u32 [[R1:%r[0-9]+]], [foo_St4x8_param_1];
  ; CHECK:       ld.param.u32 [[R2:%r[0-9]+]], [foo_St4x8_param_0];
  ; CHECK:       st.u32  [[[R1]]], [[R2]];
  ; CHECK:       ld.param.u32 [[R3:%r[0-9]+]], [foo_St4x8_param_0+4];
  ; CHECK:       st.u32  [[[R1]]+4], [[R3]];
  ; CHECK:       ld.param.u32 [[R4:%r[0-9]+]], [foo_St4x8_param_0+8];
  ; CHECK:       st.u32  [[[R1]]+8], [[R4]];
  ; CHECK:       ld.param.u32 [[R5:%r[0-9]+]], [foo_St4x8_param_0+12];
  ; CHECK:       st.u32  [[[R1]]+12], [[R5]];
  ; CHECK:       ld.param.u32 [[R6:%r[0-9]+]], [foo_St4x8_param_0+16];
  ; CHECK:       st.u32  [[[R1]]+16], [[R6]];
  ; CHECK:       ld.param.u32 [[R7:%r[0-9]+]], [foo_St4x8_param_0+20];
  ; CHECK:       st.u32  [[[R1]]+20], [[R7]];
  ; CHECK:       ld.param.u32 [[R8:%r[0-9]+]], [foo_St4x8_param_0+24];
  ; CHECK:       st.u32  [[[R1]]+24], [[R8]];
  ; CHECK:       ld.param.u32 [[R9:%r[0-9]+]], [foo_St4x8_param_0+28];
  ; CHECK:       st.u32  [[[R1]]+28], [[R9]];
  ; CHECK:       ret;
  %1 = load i32, ptr %in, align 4
  store i32 %1, ptr %ret, align 4
  %arrayidx.1 = getelementptr inbounds [8 x i32], ptr %in, i64 0, i64 1
  %2 = load i32, ptr %arrayidx.1, align 4
  %arrayidx3.1 = getelementptr inbounds [8 x i32], ptr %ret, i64 0, i64 1
  store i32 %2, ptr %arrayidx3.1, align 4
  %arrayidx.2 = getelementptr inbounds [8 x i32], ptr %in, i64 0, i64 2
  %3 = load i32, ptr %arrayidx.2, align 4
  %arrayidx3.2 = getelementptr inbounds [8 x i32], ptr %ret, i64 0, i64 2
  store i32 %3, ptr %arrayidx3.2, align 4
  %arrayidx.3 = getelementptr inbounds [8 x i32], ptr %in, i64 0, i64 3
  %4 = load i32, ptr %arrayidx.3, align 4
  %arrayidx3.3 = getelementptr inbounds [8 x i32], ptr %ret, i64 0, i64 3
  store i32 %4, ptr %arrayidx3.3, align 4
  %arrayidx.4 = getelementptr inbounds [8 x i32], ptr %in, i64 0, i64 4
  %5 = load i32, ptr %arrayidx.4, align 4
  %arrayidx3.4 = getelementptr inbounds [8 x i32], ptr %ret, i64 0, i64 4
  store i32 %5, ptr %arrayidx3.4, align 4
  %arrayidx.5 = getelementptr inbounds [8 x i32], ptr %in, i64 0, i64 5
  %6 = load i32, ptr %arrayidx.5, align 4
  %arrayidx3.5 = getelementptr inbounds [8 x i32], ptr %ret, i64 0, i64 5
  store i32 %6, ptr %arrayidx3.5, align 4
  %arrayidx.6 = getelementptr inbounds [8 x i32], ptr %in, i64 0, i64 6
  %7 = load i32, ptr %arrayidx.6, align 4
  %arrayidx3.6 = getelementptr inbounds [8 x i32], ptr %ret, i64 0, i64 6
  store i32 %7, ptr %arrayidx3.6, align 4
  %arrayidx.7 = getelementptr inbounds [8 x i32], ptr %in, i64 0, i64 7
  %8 = load i32, ptr %arrayidx.7, align 4
  %arrayidx3.7 = getelementptr inbounds [8 x i32], ptr %ret, i64 0, i64 7
  store i32 %8, ptr %arrayidx3.7, align 4
  ret void
}

define dso_local void @foo_St8x1(ptr nocapture noundef readonly byval(%struct.St8x1) align 8 %in, ptr nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func foo_St8x1(
  ; CHECK:               .param .align 8 .b8 foo_St8x1_param_0[8],
  ; CHECK:               .param .b32 foo_St8x1_param_1
  ; CHECK:       )
  ; CHECK:       ld.param.u32 [[R1:%r[0-9]+]], [foo_St8x1_param_1];
  ; CHECK:       ld.param.u64 [[RD1:%rd[0-9]+]], [foo_St8x1_param_0];
  ; CHECK:       st.u64 [[[R1]]], [[RD1]];
  ; CHECK:       ret;
  %1 = load i64, ptr %in, align 8
  store i64 %1, ptr %ret, align 8
  ret void
}

define dso_local void @foo_St8x2(ptr nocapture noundef readonly byval(%struct.St8x2) align 8 %in, ptr nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func foo_St8x2(
  ; CHECK:               .param .align 8 .b8 foo_St8x2_param_0[16],
  ; CHECK:               .param .b32 foo_St8x2_param_1
  ; CHECK:       )
  ; CHECK:       ld.param.u32 [[R1:%r[0-9]+]], [foo_St8x2_param_1];
  ; CHECK:       ld.param.u64 [[RD1:%rd[0-9]+]], [foo_St8x2_param_0];
  ; CHECK:       st.u64 [[[R1]]], [[RD1]];
  ; CHECK:       ld.param.u64 [[RD2:%rd[0-9]+]], [foo_St8x2_param_0+8];
  ; CHECK:       st.u64 [[[R1]]+8], [[RD2]];
  ; CHECK:       ret;
  %1 = load i64, ptr %in, align 8
  store i64 %1, ptr %ret, align 8
  %arrayidx.1 = getelementptr inbounds [2 x i64], ptr %in, i64 0, i64 1
  %2 = load i64, ptr %arrayidx.1, align 8
  %arrayidx3.1 = getelementptr inbounds [2 x i64], ptr %ret, i64 0, i64 1
  store i64 %2, ptr %arrayidx3.1, align 8
  ret void
}

define dso_local void @foo_St8x3(ptr nocapture noundef readonly byval(%struct.St8x3) align 8 %in, ptr nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func foo_St8x3(
  ; CHECK:               .param .align 8 .b8 foo_St8x3_param_0[24],
  ; CHECK:               .param .b32 foo_St8x3_param_1
  ; CHECK:       )
  ; CHECK:       ld.param.u32 [[R1:%r[0-9]+]], [foo_St8x3_param_1];
  ; CHECK:       ld.param.u64 [[RD1:%rd[0-9]+]], [foo_St8x3_param_0];
  ; CHECK:       st.u64 [[[R1]]], [[RD1]];
  ; CHECK:       ld.param.u64 [[RD2:%rd[0-9]+]], [foo_St8x3_param_0+8];
  ; CHECK:       st.u64 [[[R1]]+8], [[RD2]];
  ; CHECK:       ld.param.u64 [[RD3:%rd[0-9]+]], [foo_St8x3_param_0+16];
  ; CHECK:       st.u64 [[[R1]]+16], [[RD3]];
  ; CHECK:       ret;
  %1 = load i64, ptr %in, align 8
  store i64 %1, ptr %ret, align 8
  %arrayidx.1 = getelementptr inbounds [3 x i64], ptr %in, i64 0, i64 1
  %2 = load i64, ptr %arrayidx.1, align 8
  %arrayidx3.1 = getelementptr inbounds [3 x i64], ptr %ret, i64 0, i64 1
  store i64 %2, ptr %arrayidx3.1, align 8
  %arrayidx.2 = getelementptr inbounds [3 x i64], ptr %in, i64 0, i64 2
  %3 = load i64, ptr %arrayidx.2, align 8
  %arrayidx3.2 = getelementptr inbounds [3 x i64], ptr %ret, i64 0, i64 2
  store i64 %3, ptr %arrayidx3.2, align 8
  ret void
}

define dso_local void @foo_St8x4(ptr nocapture noundef readonly byval(%struct.St8x4) align 8 %in, ptr nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func foo_St8x4(
  ; CHECK:               .param .align 8 .b8 foo_St8x4_param_0[32],
  ; CHECK:               .param .b32 foo_St8x4_param_1
  ; CHECK:       )
  ; CHECK:       ld.param.u32 [[R1:%r[0-9]+]], [foo_St8x4_param_1];
  ; CHECK:       ld.param.u64 [[RD1:%rd[0-9]+]], [foo_St8x4_param_0];
  ; CHECK:       st.u64 [[[R1]]], [[RD1]];
  ; CHECK:       ld.param.u64 [[RD2:%rd[0-9]+]], [foo_St8x4_param_0+8];
  ; CHECK:       st.u64 [[[R1]]+8], [[RD2]];
  ; CHECK:       ld.param.u64 [[RD3:%rd[0-9]+]], [foo_St8x4_param_0+16];
  ; CHECK:       st.u64 [[[R1]]+16], [[RD3]];
  ; CHECK:       ld.param.u64 [[RD4:%rd[0-9]+]], [foo_St8x4_param_0+24];
  ; CHECK:       st.u64 [[[R1]]+24], [[RD4]];
  ; CHECK:       ret;
  %1 = load i64, ptr %in, align 8
  store i64 %1, ptr %ret, align 8
  %arrayidx.1 = getelementptr inbounds [4 x i64], ptr %in, i64 0, i64 1
  %2 = load i64, ptr %arrayidx.1, align 8
  %arrayidx3.1 = getelementptr inbounds [4 x i64], ptr %ret, i64 0, i64 1
  store i64 %2, ptr %arrayidx3.1, align 8
  %arrayidx.2 = getelementptr inbounds [4 x i64], ptr %in, i64 0, i64 2
  %3 = load i64, ptr %arrayidx.2, align 8
  %arrayidx3.2 = getelementptr inbounds [4 x i64], ptr %ret, i64 0, i64 2
  store i64 %3, ptr %arrayidx3.2, align 8
  %arrayidx.3 = getelementptr inbounds [4 x i64], ptr %in, i64 0, i64 3
  %4 = load i64, ptr %arrayidx.3, align 8
  %arrayidx3.3 = getelementptr inbounds [4 x i64], ptr %ret, i64 0, i64 3
  store i64 %4, ptr %arrayidx3.3, align 8
  ret void
}
