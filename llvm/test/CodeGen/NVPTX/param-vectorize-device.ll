; RUN: llc < %s -mtriple=nvptx-unknown-unknown | FileCheck %s
;
; Check that parameters of a __device__ function with private or internal
; linkage called from a __global__ (kernel) function get increased alignment,
; and additional vectorization is performed on loads/stores with that
; parameters.
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
; #define DECLARE_CALLEE(StName)                                      \
; static __device__  __attribute__((noinline))                        \
; struct StName callee_##StName(struct StName in) {                   \
;   struct StName ret;                                                \
;   const unsigned size = sizeof(ret.field) / sizeof(*ret.field);     \
;   for (unsigned i = 0; i != size; ++i)                              \
;     ret.field[i] = in.field[i];                                     \
;   return ret;                                                       \
; }                                                                   \

; #define DECLARE_CALLER(StName)                                      \
; __global__                                                          \
; void caller_##StName(struct StName in, struct StName* ret)          \
; {                                                                   \
;   *ret = callee_##StName(in);                                       \
; }                                                                   \
;
; #define DECLARE_CALL(StName)  \
;     DECLARE_CALLEE(StName)    \
;     DECLARE_CALLER(StName)    \
;
; DECLARE_CALL(St4x1)
; DECLARE_CALL(St4x2)
; DECLARE_CALL(St4x3)
; DECLARE_CALL(St4x4)
; DECLARE_CALL(St4x5)
; DECLARE_CALL(St4x6)
; DECLARE_CALL(St4x7)
; DECLARE_CALL(St4x8)
; DECLARE_CALL(St8x1)
; DECLARE_CALL(St8x2)
; DECLARE_CALL(St8x3)
; DECLARE_CALL(St8x4)
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

; Section 1 - checking that:
; - function argument (including retval) vectorization is done with internal linkage;
; - caller and callee specify correct alignment for callee's params.

define dso_local void @caller_St4x1(%struct.St4x1* nocapture noundef readonly byval(%struct.St4x1) align 4 %in, %struct.St4x1* nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func caller_St4x1(
  ; CHECK:               .param .align 4 .b8 caller_St4x1_param_0[4],
  ; CHECK:               .param .b32 caller_St4x1_param_1
  ; CHECK:       )
  ; CHECK:       .param .b32 param0;
  ; CHECK:       st.param.b32 [param0+0], {{%r[0-9]+}};
  ; CHECK:       .param .align 16 .b8 retval0[4];
  ; CHECK:       call.uni (retval0),
  ; CHECK-NEXT:  callee_St4x1,
  ; CHECK-NEXT:  (
  ; CHECK-NEXT:  param0
  ; CHECK-NEXT:  );
  ; CHECK:       ld.param.b32 {{%r[0-9]+}}, [retval0+0];
  %1 = getelementptr inbounds %struct.St4x1, %struct.St4x1* %in, i64 0, i32 0, i64 0
  %2 = load i32, i32* %1, align 4
  %call = tail call fastcc [1 x i32] @callee_St4x1(i32 %2)
  %.fca.0.extract = extractvalue [1 x i32] %call, 0
  %ref.tmp.sroa.0.0..sroa_idx = getelementptr inbounds %struct.St4x1, %struct.St4x1* %ret, i64 0, i32 0, i64 0
  store i32 %.fca.0.extract, i32* %ref.tmp.sroa.0.0..sroa_idx, align 4
  ret void
}

define internal fastcc [1 x i32] @callee_St4x1(i32 %in.0.val) {
  ; CHECK:       .func  (.param .align 16 .b8 func_retval0[4])
  ; CHECK-LABEL: callee_St4x1(
  ; CHECK-NEXT:  .param .b32 callee_St4x1_param_0
  ; CHECK:       ld.param.u32 [[R1:%r[0-9]+]], [callee_St4x1_param_0];
  ; CHECK:       st.param.b32 [func_retval0+0], [[R1]];
  ; CHECK-NEXT:  ret;
  %oldret = insertvalue [1 x i32] poison, i32 %in.0.val, 0
  ret [1 x i32] %oldret
}

define dso_local void @caller_St4x2(%struct.St4x2* nocapture noundef readonly byval(%struct.St4x2) align 4 %in, %struct.St4x2* nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func caller_St4x2(
  ; CHECK:               .param .align 4 .b8 caller_St4x2_param_0[8],
  ; CHECK:               .param .b32 caller_St4x2_param_1
  ; CHECK:       )
  ; CHECK:       .param .align 16 .b8 param0[8];
  ; CHECK:       st.param.v2.b32 [param0+0], {{{%r[0-9]+}}, {{%r[0-9]+}}};
  ; CHECK:       .param .align 16 .b8 retval0[8];
  ; CHECK:       call.uni (retval0),
  ; CHECK-NEXT:  callee_St4x2,
  ; CHECK-NEXT:  (
  ; CHECK-NEXT:  param0
  ; CHECK-NEXT:  );
  ; CHECK:       ld.param.v2.b32 {{{%r[0-9]+}}, {{%r[0-9]+}}}, [retval0+0];
  %agg.tmp = alloca i64, align 8
  %tmpcast = bitcast i64* %agg.tmp to %struct.St4x2*
  %1 = bitcast %struct.St4x2* %in to i64*
  %2 = load i64, i64* %1, align 4
  store i64 %2, i64* %agg.tmp, align 8
  %call = tail call fastcc [2 x i32] @callee_St4x2(%struct.St4x2* noundef nonnull byval(%struct.St4x2) align 4 %tmpcast)
  %.fca.0.extract = extractvalue [2 x i32] %call, 0
  %.fca.1.extract = extractvalue [2 x i32] %call, 1
  %ref.tmp.sroa.0.0..sroa_idx = getelementptr inbounds %struct.St4x2, %struct.St4x2* %ret, i64 0, i32 0, i64 0
  store i32 %.fca.0.extract, i32* %ref.tmp.sroa.0.0..sroa_idx, align 4
  %ref.tmp.sroa.4.0..sroa_idx3 = getelementptr inbounds %struct.St4x2, %struct.St4x2* %ret, i64 0, i32 0, i64 1
  store i32 %.fca.1.extract, i32* %ref.tmp.sroa.4.0..sroa_idx3, align 4
  ret void
}

define internal fastcc [2 x i32] @callee_St4x2(%struct.St4x2* nocapture noundef readonly byval(%struct.St4x2) align 4 %in) {
  ; CHECK:       .func  (.param .align 16 .b8 func_retval0[8])
  ; CHECK-LABEL: callee_St4x2(
  ; CHECK-NEXT:  .param .align 16 .b8 callee_St4x2_param_0[8]
  ; CHECK:       ld.param.v2.u32 {[[R1:%r[0-9]+]], [[R2:%r[0-9]+]]}, [callee_St4x2_param_0];
  ; CHECK:       st.param.v2.b32 [func_retval0+0], {[[R1]], [[R2]]};
  ; CHECK-NEXT:  ret;
  %arrayidx = getelementptr inbounds %struct.St4x2, %struct.St4x2* %in, i64 0, i32 0, i64 0
  %1 = load i32, i32* %arrayidx, align 4
  %arrayidx.1 = getelementptr inbounds %struct.St4x2, %struct.St4x2* %in, i64 0, i32 0, i64 1
  %2 = load i32, i32* %arrayidx.1, align 4
  %3 = insertvalue [2 x i32] poison, i32 %1, 0
  %oldret = insertvalue [2 x i32] %3, i32 %2, 1
  ret [2 x i32] %oldret
}

define dso_local void @caller_St4x3(%struct.St4x3* nocapture noundef readonly byval(%struct.St4x3) align 4 %in, %struct.St4x3* nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func caller_St4x3(
  ; CHECK:               .param .align 4 .b8 caller_St4x3_param_0[12],
  ; CHECK:               .param .b32 caller_St4x3_param_1
  ; CHECK:       )
  ; CHECK:       .param .align 16 .b8 param0[12];
  ; CHECK:       st.param.v2.b32 [param0+0], {{{%r[0-9]+}}, {{%r[0-9]+}}};
  ; CHECK:       st.param.b32    [param0+8], {{%r[0-9]+}};
  ; CHECK:       .param .align 16 .b8 retval0[12];
  ; CHECK:       call.uni (retval0),
  ; CHECK-NEXT:  callee_St4x3,
  ; CHECK-NEXT:  (
  ; CHECK-NEXT:  param0
  ; CHECK-NEXT:  );
  ; CHECK:       ld.param.v2.b32 {{{%r[0-9]+}}, {{%r[0-9]+}}}, [retval0+0];
  ; CHECK:       ld.param.b32    {{%r[0-9]+}},  [retval0+8];
  %call = tail call fastcc [3 x i32] @callee_St4x3(%struct.St4x3* noundef nonnull byval(%struct.St4x3) align 4 %in)
  %.fca.0.extract = extractvalue [3 x i32] %call, 0
  %.fca.1.extract = extractvalue [3 x i32] %call, 1
  %.fca.2.extract = extractvalue [3 x i32] %call, 2
  %ref.tmp.sroa.0.0..sroa_idx = getelementptr inbounds %struct.St4x3, %struct.St4x3* %ret, i64 0, i32 0, i64 0
  store i32 %.fca.0.extract, i32* %ref.tmp.sroa.0.0..sroa_idx, align 4
  %ref.tmp.sroa.4.0..sroa_idx2 = getelementptr inbounds %struct.St4x3, %struct.St4x3* %ret, i64 0, i32 0, i64 1
  store i32 %.fca.1.extract, i32* %ref.tmp.sroa.4.0..sroa_idx2, align 4
  %ref.tmp.sroa.5.0..sroa_idx4 = getelementptr inbounds %struct.St4x3, %struct.St4x3* %ret, i64 0, i32 0, i64 2
  store i32 %.fca.2.extract, i32* %ref.tmp.sroa.5.0..sroa_idx4, align 4
  ret void
}


define internal fastcc [3 x i32] @callee_St4x3(%struct.St4x3* nocapture noundef readonly byval(%struct.St4x3) align 4 %in) {
  ; CHECK:       .func  (.param .align 16 .b8 func_retval0[12])
  ; CHECK-LABEL: callee_St4x3(
  ; CHECK-NEXT:  .param .align 16 .b8 callee_St4x3_param_0[12]
  ; CHECK:       ld.param.v2.u32 {[[R1:%r[0-9]+]], [[R2:%r[0-9]+]]}, [callee_St4x3_param_0];
  ; CHECK:       ld.param.u32    [[R3:%r[0-9]+]],  [callee_St4x3_param_0+8];
  ; CHECK:       st.param.v2.b32 [func_retval0+0], {[[R1]], [[R2]]};
  ; CHECK:       st.param.b32    [func_retval0+8], [[R3]];
  ; CHECK-NEXT:  ret;
  %arrayidx = getelementptr inbounds %struct.St4x3, %struct.St4x3* %in, i64 0, i32 0, i64 0
  %1 = load i32, i32* %arrayidx, align 4
  %arrayidx.1 = getelementptr inbounds %struct.St4x3, %struct.St4x3* %in, i64 0, i32 0, i64 1
  %2 = load i32, i32* %arrayidx.1, align 4
  %arrayidx.2 = getelementptr inbounds %struct.St4x3, %struct.St4x3* %in, i64 0, i32 0, i64 2
  %3 = load i32, i32* %arrayidx.2, align 4
  %4 = insertvalue [3 x i32] poison, i32 %1, 0
  %5 = insertvalue [3 x i32] %4, i32 %2, 1
  %oldret = insertvalue [3 x i32] %5, i32 %3, 2
  ret [3 x i32] %oldret
}


define dso_local void @caller_St4x4(%struct.St4x4* nocapture noundef readonly byval(%struct.St4x4) align 4 %in, %struct.St4x4* nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func caller_St4x4(
  ; CHECK:               .param .align 4 .b8 caller_St4x4_param_0[16],
  ; CHECK:               .param .b32 caller_St4x4_param_1
  ; CHECK:       )
  ; CHECK:       .param .align 16 .b8 param0[16];
  ; CHECK:       st.param.v4.b32 [param0+0], {{{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}};
  ; CHECK:       .param .align 16 .b8 retval0[16];
  ; CHECK:       call.uni (retval0),
  ; CHECK-NEXT:  callee_St4x4,
  ; CHECK-NEXT:  (
  ; CHECK-NEXT:  param0
  ; CHECK-NEXT:  );
  ; CHECK:       ld.param.v4.b32 {{{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}}, [retval0+0];
  %call = tail call fastcc [4 x i32] @callee_St4x4(%struct.St4x4* noundef nonnull byval(%struct.St4x4) align 4 %in)
  %.fca.0.extract = extractvalue [4 x i32] %call, 0
  %.fca.1.extract = extractvalue [4 x i32] %call, 1
  %.fca.2.extract = extractvalue [4 x i32] %call, 2
  %.fca.3.extract = extractvalue [4 x i32] %call, 3
  %ref.tmp.sroa.0.0..sroa_idx = getelementptr inbounds %struct.St4x4, %struct.St4x4* %ret, i64 0, i32 0, i64 0
  store i32 %.fca.0.extract, i32* %ref.tmp.sroa.0.0..sroa_idx, align 4
  %ref.tmp.sroa.4.0..sroa_idx3 = getelementptr inbounds %struct.St4x4, %struct.St4x4* %ret, i64 0, i32 0, i64 1
  store i32 %.fca.1.extract, i32* %ref.tmp.sroa.4.0..sroa_idx3, align 4
  %ref.tmp.sroa.5.0..sroa_idx5 = getelementptr inbounds %struct.St4x4, %struct.St4x4* %ret, i64 0, i32 0, i64 2
  store i32 %.fca.2.extract, i32* %ref.tmp.sroa.5.0..sroa_idx5, align 4
  %ref.tmp.sroa.6.0..sroa_idx7 = getelementptr inbounds %struct.St4x4, %struct.St4x4* %ret, i64 0, i32 0, i64 3
  store i32 %.fca.3.extract, i32* %ref.tmp.sroa.6.0..sroa_idx7, align 4
  ret void
}


define internal fastcc [4 x i32] @callee_St4x4(%struct.St4x4* nocapture noundef readonly byval(%struct.St4x4) align 4 %in) {
  ; CHECK:       .func  (.param .align 16 .b8 func_retval0[16])
  ; CHECK-LABEL: callee_St4x4(
  ; CHECK-NEXT:  .param .align 16 .b8 callee_St4x4_param_0[16]
  ; CHECK:       ld.param.v4.u32 {[[R1:%r[0-9]+]], [[R2:%r[0-9]+]], [[R3:%r[0-9]+]], [[R4:%r[0-9]+]]}, [callee_St4x4_param_0];
  ; CHECK:       st.param.v4.b32 [func_retval0+0], {[[R1]], [[R2]], [[R3]], [[R4]]};
  ; CHECK-NEXT:  ret;
  %arrayidx = getelementptr inbounds %struct.St4x4, %struct.St4x4* %in, i64 0, i32 0, i64 0
  %1 = load i32, i32* %arrayidx, align 4
  %arrayidx.1 = getelementptr inbounds %struct.St4x4, %struct.St4x4* %in, i64 0, i32 0, i64 1
  %2 = load i32, i32* %arrayidx.1, align 4
  %arrayidx.2 = getelementptr inbounds %struct.St4x4, %struct.St4x4* %in, i64 0, i32 0, i64 2
  %3 = load i32, i32* %arrayidx.2, align 4
  %arrayidx.3 = getelementptr inbounds %struct.St4x4, %struct.St4x4* %in, i64 0, i32 0, i64 3
  %4 = load i32, i32* %arrayidx.3, align 4
  %5 = insertvalue [4 x i32] poison, i32 %1, 0
  %6 = insertvalue [4 x i32] %5, i32 %2, 1
  %7 = insertvalue [4 x i32] %6, i32 %3, 2
  %oldret = insertvalue [4 x i32] %7, i32 %4, 3
  ret [4 x i32] %oldret
}


define dso_local void @caller_St4x5(%struct.St4x5* nocapture noundef readonly byval(%struct.St4x5) align 4 %in, %struct.St4x5* nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func caller_St4x5(
  ; CHECK:               .param .align 4 .b8 caller_St4x5_param_0[20],
  ; CHECK:               .param .b32 caller_St4x5_param_1
  ; CHECK:       )
  ; CHECK:       .param .align 16 .b8 param0[20];
  ; CHECK:       st.param.v4.b32 [param0+0],  {{{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}};
  ; CHECK:       st.param.b32    [param0+16], {{%r[0-9]+}};
  ; CHECK:       .param .align 16 .b8 retval0[20];
  ; CHECK:       call.uni (retval0),
  ; CHECK-NEXT:  callee_St4x5,
  ; CHECK-NEXT:  (
  ; CHECK-NEXT:  param0
  ; CHECK-NEXT:  );
  ; CHECK:       ld.param.v4.b32 {{{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}}, [retval0+0];
  ; CHECK:       ld.param.b32    {{%r[0-9]+}},  [retval0+16];
  %call = tail call fastcc [5 x i32] @callee_St4x5(%struct.St4x5* noundef nonnull byval(%struct.St4x5) align 4 %in)
  %.fca.0.extract = extractvalue [5 x i32] %call, 0
  %.fca.1.extract = extractvalue [5 x i32] %call, 1
  %.fca.2.extract = extractvalue [5 x i32] %call, 2
  %.fca.3.extract = extractvalue [5 x i32] %call, 3
  %.fca.4.extract = extractvalue [5 x i32] %call, 4
  %ref.tmp.sroa.0.0..sroa_idx = getelementptr inbounds %struct.St4x5, %struct.St4x5* %ret, i64 0, i32 0, i64 0
  store i32 %.fca.0.extract, i32* %ref.tmp.sroa.0.0..sroa_idx, align 4
  %ref.tmp.sroa.4.0..sroa_idx3 = getelementptr inbounds %struct.St4x5, %struct.St4x5* %ret, i64 0, i32 0, i64 1
  store i32 %.fca.1.extract, i32* %ref.tmp.sroa.4.0..sroa_idx3, align 4
  %ref.tmp.sroa.5.0..sroa_idx5 = getelementptr inbounds %struct.St4x5, %struct.St4x5* %ret, i64 0, i32 0, i64 2
  store i32 %.fca.2.extract, i32* %ref.tmp.sroa.5.0..sroa_idx5, align 4
  %ref.tmp.sroa.6.0..sroa_idx7 = getelementptr inbounds %struct.St4x5, %struct.St4x5* %ret, i64 0, i32 0, i64 3
  store i32 %.fca.3.extract, i32* %ref.tmp.sroa.6.0..sroa_idx7, align 4
  %ref.tmp.sroa.7.0..sroa_idx9 = getelementptr inbounds %struct.St4x5, %struct.St4x5* %ret, i64 0, i32 0, i64 4
  store i32 %.fca.4.extract, i32* %ref.tmp.sroa.7.0..sroa_idx9, align 4
  ret void
}


define internal fastcc [5 x i32] @callee_St4x5(%struct.St4x5* nocapture noundef readonly byval(%struct.St4x5) align 4 %in) {
  ; CHECK:       .func  (.param .align 16 .b8 func_retval0[20])
  ; CHECK-LABEL: callee_St4x5(
  ; CHECK-NEXT:  .param .align 16 .b8 callee_St4x5_param_0[20]
  ; CHECK:       ld.param.v4.u32 {[[R1:%r[0-9]+]], [[R2:%r[0-9]+]], [[R3:%r[0-9]+]], [[R4:%r[0-9]+]]}, [callee_St4x5_param_0];
  ; CHECK:       ld.param.u32    [[R5:%r[0-9]+]],   [callee_St4x5_param_0+16];
  ; CHECK:       st.param.v4.b32 [func_retval0+0],  {[[R1]], [[R2]], [[R3]], [[R4]]};
  ; CHECK:       st.param.b32    [func_retval0+16], [[R5]];
  ; CHECK-NEXT:  ret;
  %arrayidx = getelementptr inbounds %struct.St4x5, %struct.St4x5* %in, i64 0, i32 0, i64 0
  %1 = load i32, i32* %arrayidx, align 4
  %arrayidx.1 = getelementptr inbounds %struct.St4x5, %struct.St4x5* %in, i64 0, i32 0, i64 1
  %2 = load i32, i32* %arrayidx.1, align 4
  %arrayidx.2 = getelementptr inbounds %struct.St4x5, %struct.St4x5* %in, i64 0, i32 0, i64 2
  %3 = load i32, i32* %arrayidx.2, align 4
  %arrayidx.3 = getelementptr inbounds %struct.St4x5, %struct.St4x5* %in, i64 0, i32 0, i64 3
  %4 = load i32, i32* %arrayidx.3, align 4
  %arrayidx.4 = getelementptr inbounds %struct.St4x5, %struct.St4x5* %in, i64 0, i32 0, i64 4
  %5 = load i32, i32* %arrayidx.4, align 4
  %6 = insertvalue [5 x i32] poison, i32 %1, 0
  %7 = insertvalue [5 x i32] %6, i32 %2, 1
  %8 = insertvalue [5 x i32] %7, i32 %3, 2
  %9 = insertvalue [5 x i32] %8, i32 %4, 3
  %oldret = insertvalue [5 x i32] %9, i32 %5, 4
  ret [5 x i32] %oldret
}


define dso_local void @caller_St4x6(%struct.St4x6* nocapture noundef readonly byval(%struct.St4x6) align 4 %in, %struct.St4x6* nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func caller_St4x6(
  ; CHECK:               .param .align 4 .b8 caller_St4x6_param_0[24],
  ; CHECK:               .param .b32 caller_St4x6_param_1
  ; CHECK:       )
  ; CHECK:       .param .align 16 .b8 param0[24];
  ; CHECK:       st.param.v4.b32 [param0+0],  {{{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}};
  ; CHECK:       st.param.v2.b32 [param0+16], {{{%r[0-9]+}}, {{%r[0-9]+}}};
  ; CHECK:       .param .align 16 .b8 retval0[24];
  ; CHECK:       call.uni (retval0),
  ; CHECK-NEXT:  callee_St4x6,
  ; CHECK-NEXT:  (
  ; CHECK-NEXT:  param0
  ; CHECK-NEXT:  );
  ; CHECK:       ld.param.v4.b32 {{{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}}, [retval0+0];
  ; CHECK:       ld.param.v2.b32 {{{%r[0-9]+}}, {{%r[0-9]+}}}, [retval0+16];
  %call = tail call fastcc [6 x i32] @callee_St4x6(%struct.St4x6* noundef nonnull byval(%struct.St4x6) align 4 %in)
  %.fca.0.extract = extractvalue [6 x i32] %call, 0
  %.fca.1.extract = extractvalue [6 x i32] %call, 1
  %.fca.2.extract = extractvalue [6 x i32] %call, 2
  %.fca.3.extract = extractvalue [6 x i32] %call, 3
  %.fca.4.extract = extractvalue [6 x i32] %call, 4
  %.fca.5.extract = extractvalue [6 x i32] %call, 5
  %ref.tmp.sroa.0.0..sroa_idx = getelementptr inbounds %struct.St4x6, %struct.St4x6* %ret, i64 0, i32 0, i64 0
  store i32 %.fca.0.extract, i32* %ref.tmp.sroa.0.0..sroa_idx, align 4
  %ref.tmp.sroa.4.0..sroa_idx2 = getelementptr inbounds %struct.St4x6, %struct.St4x6* %ret, i64 0, i32 0, i64 1
  store i32 %.fca.1.extract, i32* %ref.tmp.sroa.4.0..sroa_idx2, align 4
  %ref.tmp.sroa.5.0..sroa_idx4 = getelementptr inbounds %struct.St4x6, %struct.St4x6* %ret, i64 0, i32 0, i64 2
  store i32 %.fca.2.extract, i32* %ref.tmp.sroa.5.0..sroa_idx4, align 4
  %ref.tmp.sroa.6.0..sroa_idx6 = getelementptr inbounds %struct.St4x6, %struct.St4x6* %ret, i64 0, i32 0, i64 3
  store i32 %.fca.3.extract, i32* %ref.tmp.sroa.6.0..sroa_idx6, align 4
  %ref.tmp.sroa.7.0..sroa_idx8 = getelementptr inbounds %struct.St4x6, %struct.St4x6* %ret, i64 0, i32 0, i64 4
  store i32 %.fca.4.extract, i32* %ref.tmp.sroa.7.0..sroa_idx8, align 4
  %ref.tmp.sroa.8.0..sroa_idx10 = getelementptr inbounds %struct.St4x6, %struct.St4x6* %ret, i64 0, i32 0, i64 5
  store i32 %.fca.5.extract, i32* %ref.tmp.sroa.8.0..sroa_idx10, align 4
  ret void
}


define internal fastcc [6 x i32] @callee_St4x6(%struct.St4x6* nocapture noundef readonly byval(%struct.St4x6) align 4 %in) {
  ; CHECK:       .func  (.param .align 16 .b8 func_retval0[24])
  ; CHECK-LABEL: callee_St4x6(
  ; CHECK-NEXT:  .param .align 16 .b8 callee_St4x6_param_0[24]
  ; CHECK:       ld.param.v4.u32 {[[R1:%r[0-9]+]], [[R2:%r[0-9]+]], [[R3:%r[0-9]+]], [[R4:%r[0-9]+]]}, [callee_St4x6_param_0];
  ; CHECK:       ld.param.v2.u32 {[[R5:%r[0-9]+]],  [[R6:%r[0-9]+]]}, [callee_St4x6_param_0+16];
  ; CHECK:       st.param.v4.b32 [func_retval0+0],  {[[R1]], [[R2]], [[R3]], [[R4]]};
  ; CHECK:       st.param.v2.b32 [func_retval0+16], {[[R5]], [[R6]]};
  ; CHECK-NEXT:  ret;
  %arrayidx = getelementptr inbounds %struct.St4x6, %struct.St4x6* %in, i64 0, i32 0, i64 0
  %1 = load i32, i32* %arrayidx, align 4
  %arrayidx.1 = getelementptr inbounds %struct.St4x6, %struct.St4x6* %in, i64 0, i32 0, i64 1
  %2 = load i32, i32* %arrayidx.1, align 4
  %arrayidx.2 = getelementptr inbounds %struct.St4x6, %struct.St4x6* %in, i64 0, i32 0, i64 2
  %3 = load i32, i32* %arrayidx.2, align 4
  %arrayidx.3 = getelementptr inbounds %struct.St4x6, %struct.St4x6* %in, i64 0, i32 0, i64 3
  %4 = load i32, i32* %arrayidx.3, align 4
  %arrayidx.4 = getelementptr inbounds %struct.St4x6, %struct.St4x6* %in, i64 0, i32 0, i64 4
  %5 = load i32, i32* %arrayidx.4, align 4
  %arrayidx.5 = getelementptr inbounds %struct.St4x6, %struct.St4x6* %in, i64 0, i32 0, i64 5
  %6 = load i32, i32* %arrayidx.5, align 4
  %7 = insertvalue [6 x i32] poison, i32 %1, 0
  %8 = insertvalue [6 x i32] %7, i32 %2, 1
  %9 = insertvalue [6 x i32] %8, i32 %3, 2
  %10 = insertvalue [6 x i32] %9, i32 %4, 3
  %11 = insertvalue [6 x i32] %10, i32 %5, 4
  %oldret = insertvalue [6 x i32] %11, i32 %6, 5
  ret [6 x i32] %oldret
}


define dso_local void @caller_St4x7(%struct.St4x7* nocapture noundef readonly byval(%struct.St4x7) align 4 %in, %struct.St4x7* nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func caller_St4x7(
  ; CHECK:               .param .align 4 .b8 caller_St4x7_param_0[28],
  ; CHECK:               .param .b32 caller_St4x7_param_1
  ; CHECK:       )
  ; CHECK:       .param .align 16 .b8 param0[28];
  ; CHECK:       st.param.v4.b32 [param0+0],  {{{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}};
  ; CHECK:       st.param.v2.b32 [param0+16], {{{%r[0-9]+}}, {{%r[0-9]+}}};
  ; CHECK:       st.param.b32    [param0+24], {{%r[0-9]+}};
  ; CHECK:       .param .align 16 .b8 retval0[28];
  ; CHECK:       call.uni (retval0),
  ; CHECK-NEXT:  callee_St4x7,
  ; CHECK-NEXT:  (
  ; CHECK-NEXT:  param0
  ; CHECK-NEXT:  );
  ; CHECK:       ld.param.v4.b32 {{{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}}, [retval0+0];
  ; CHECK:       ld.param.v2.b32 {{{%r[0-9]+}}, {{%r[0-9]+}}}, [retval0+16];
  ; CHECK:       ld.param.b32    {{%r[0-9]+}}, [retval0+24];
  %call = tail call fastcc [7 x i32] @callee_St4x7(%struct.St4x7* noundef nonnull byval(%struct.St4x7) align 4 %in)
  %.fca.0.extract = extractvalue [7 x i32] %call, 0
  %.fca.1.extract = extractvalue [7 x i32] %call, 1
  %.fca.2.extract = extractvalue [7 x i32] %call, 2
  %.fca.3.extract = extractvalue [7 x i32] %call, 3
  %.fca.4.extract = extractvalue [7 x i32] %call, 4
  %.fca.5.extract = extractvalue [7 x i32] %call, 5
  %.fca.6.extract = extractvalue [7 x i32] %call, 6
  %ref.tmp.sroa.0.0..sroa_idx = getelementptr inbounds %struct.St4x7, %struct.St4x7* %ret, i64 0, i32 0, i64 0
  store i32 %.fca.0.extract, i32* %ref.tmp.sroa.0.0..sroa_idx, align 4
  %ref.tmp.sroa.4.0..sroa_idx2 = getelementptr inbounds %struct.St4x7, %struct.St4x7* %ret, i64 0, i32 0, i64 1
  store i32 %.fca.1.extract, i32* %ref.tmp.sroa.4.0..sroa_idx2, align 4
  %ref.tmp.sroa.5.0..sroa_idx4 = getelementptr inbounds %struct.St4x7, %struct.St4x7* %ret, i64 0, i32 0, i64 2
  store i32 %.fca.2.extract, i32* %ref.tmp.sroa.5.0..sroa_idx4, align 4
  %ref.tmp.sroa.6.0..sroa_idx6 = getelementptr inbounds %struct.St4x7, %struct.St4x7* %ret, i64 0, i32 0, i64 3
  store i32 %.fca.3.extract, i32* %ref.tmp.sroa.6.0..sroa_idx6, align 4
  %ref.tmp.sroa.7.0..sroa_idx8 = getelementptr inbounds %struct.St4x7, %struct.St4x7* %ret, i64 0, i32 0, i64 4
  store i32 %.fca.4.extract, i32* %ref.tmp.sroa.7.0..sroa_idx8, align 4
  %ref.tmp.sroa.8.0..sroa_idx10 = getelementptr inbounds %struct.St4x7, %struct.St4x7* %ret, i64 0, i32 0, i64 5
  store i32 %.fca.5.extract, i32* %ref.tmp.sroa.8.0..sroa_idx10, align 4
  %ref.tmp.sroa.9.0..sroa_idx12 = getelementptr inbounds %struct.St4x7, %struct.St4x7* %ret, i64 0, i32 0, i64 6
  store i32 %.fca.6.extract, i32* %ref.tmp.sroa.9.0..sroa_idx12, align 4
  ret void
}


define internal fastcc [7 x i32] @callee_St4x7(%struct.St4x7* nocapture noundef readonly byval(%struct.St4x7) align 4 %in) {
  ; CHECK:       .func  (.param .align 16 .b8 func_retval0[28])
  ; CHECK-LABEL: callee_St4x7(
  ; CHECK-NEXT:  .param .align 16 .b8 callee_St4x7_param_0[28]
  ; CHECK:       ld.param.v4.u32 {[[R1:%r[0-9]+]], [[R2:%r[0-9]+]], [[R3:%r[0-9]+]], [[R4:%r[0-9]+]]}, [callee_St4x7_param_0];
  ; CHECK:       ld.param.v2.u32 {[[R5:%r[0-9]+]],  [[R6:%r[0-9]+]]}, [callee_St4x7_param_0+16];
  ; CHECK:       ld.param.u32    [[R7:%r[0-9]+]],   [callee_St4x7_param_0+24];
  ; CHECK:       st.param.v4.b32 [func_retval0+0],  {[[R1]], [[R2]], [[R3]], [[R4]]};
  ; CHECK:       st.param.v2.b32 [func_retval0+16], {[[R5]], [[R6]]};
  ; CHECK:       st.param.b32    [func_retval0+24], [[R7]];
  ; CHECK-NEXT:  ret;
  %arrayidx = getelementptr inbounds %struct.St4x7, %struct.St4x7* %in, i64 0, i32 0, i64 0
  %1 = load i32, i32* %arrayidx, align 4
  %arrayidx.1 = getelementptr inbounds %struct.St4x7, %struct.St4x7* %in, i64 0, i32 0, i64 1
  %2 = load i32, i32* %arrayidx.1, align 4
  %arrayidx.2 = getelementptr inbounds %struct.St4x7, %struct.St4x7* %in, i64 0, i32 0, i64 2
  %3 = load i32, i32* %arrayidx.2, align 4
  %arrayidx.3 = getelementptr inbounds %struct.St4x7, %struct.St4x7* %in, i64 0, i32 0, i64 3
  %4 = load i32, i32* %arrayidx.3, align 4
  %arrayidx.4 = getelementptr inbounds %struct.St4x7, %struct.St4x7* %in, i64 0, i32 0, i64 4
  %5 = load i32, i32* %arrayidx.4, align 4
  %arrayidx.5 = getelementptr inbounds %struct.St4x7, %struct.St4x7* %in, i64 0, i32 0, i64 5
  %6 = load i32, i32* %arrayidx.5, align 4
  %arrayidx.6 = getelementptr inbounds %struct.St4x7, %struct.St4x7* %in, i64 0, i32 0, i64 6
  %7 = load i32, i32* %arrayidx.6, align 4
  %8 = insertvalue [7 x i32] poison, i32 %1, 0
  %9 = insertvalue [7 x i32] %8, i32 %2, 1
  %10 = insertvalue [7 x i32] %9, i32 %3, 2
  %11 = insertvalue [7 x i32] %10, i32 %4, 3
  %12 = insertvalue [7 x i32] %11, i32 %5, 4
  %13 = insertvalue [7 x i32] %12, i32 %6, 5
  %oldret = insertvalue [7 x i32] %13, i32 %7, 6
  ret [7 x i32] %oldret
}


define dso_local void @caller_St4x8(%struct.St4x8* nocapture noundef readonly byval(%struct.St4x8) align 4 %in, %struct.St4x8* nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func caller_St4x8(
  ; CHECK:               .param .align 4 .b8 caller_St4x8_param_0[32],
  ; CHECK:               .param .b32 caller_St4x8_param_1
  ; CHECK:       )
  ; CHECK:       .param .align 16 .b8 param0[32];
  ; CHECK:       st.param.v4.b32 [param0+0],  {{{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}};
  ; CHECK:       st.param.v4.b32 [param0+16], {{{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}};
  ; CHECK:       .param .align 16 .b8 retval0[32];
  ; CHECK:       call.uni (retval0),
  ; CHECK-NEXT:  callee_St4x8,
  ; CHECK-NEXT:  (
  ; CHECK-NEXT:  param0
  ; CHECK-NEXT:  );
  ; CHECK:       ld.param.v4.b32 {{{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}}, [retval0+0];
  ; CHECK:       ld.param.v4.b32 {{{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}}, [retval0+16];
  %call = tail call fastcc [8 x i32] @callee_St4x8(%struct.St4x8* noundef nonnull byval(%struct.St4x8) align 4 %in)
  %.fca.0.extract = extractvalue [8 x i32] %call, 0
  %.fca.1.extract = extractvalue [8 x i32] %call, 1
  %.fca.2.extract = extractvalue [8 x i32] %call, 2
  %.fca.3.extract = extractvalue [8 x i32] %call, 3
  %.fca.4.extract = extractvalue [8 x i32] %call, 4
  %.fca.5.extract = extractvalue [8 x i32] %call, 5
  %.fca.6.extract = extractvalue [8 x i32] %call, 6
  %.fca.7.extract = extractvalue [8 x i32] %call, 7
  %ref.tmp.sroa.0.0..sroa_idx = getelementptr inbounds %struct.St4x8, %struct.St4x8* %ret, i64 0, i32 0, i64 0
  store i32 %.fca.0.extract, i32* %ref.tmp.sroa.0.0..sroa_idx, align 4
  %ref.tmp.sroa.4.0..sroa_idx2 = getelementptr inbounds %struct.St4x8, %struct.St4x8* %ret, i64 0, i32 0, i64 1
  store i32 %.fca.1.extract, i32* %ref.tmp.sroa.4.0..sroa_idx2, align 4
  %ref.tmp.sroa.5.0..sroa_idx4 = getelementptr inbounds %struct.St4x8, %struct.St4x8* %ret, i64 0, i32 0, i64 2
  store i32 %.fca.2.extract, i32* %ref.tmp.sroa.5.0..sroa_idx4, align 4
  %ref.tmp.sroa.6.0..sroa_idx6 = getelementptr inbounds %struct.St4x8, %struct.St4x8* %ret, i64 0, i32 0, i64 3
  store i32 %.fca.3.extract, i32* %ref.tmp.sroa.6.0..sroa_idx6, align 4
  %ref.tmp.sroa.7.0..sroa_idx8 = getelementptr inbounds %struct.St4x8, %struct.St4x8* %ret, i64 0, i32 0, i64 4
  store i32 %.fca.4.extract, i32* %ref.tmp.sroa.7.0..sroa_idx8, align 4
  %ref.tmp.sroa.8.0..sroa_idx10 = getelementptr inbounds %struct.St4x8, %struct.St4x8* %ret, i64 0, i32 0, i64 5
  store i32 %.fca.5.extract, i32* %ref.tmp.sroa.8.0..sroa_idx10, align 4
  %ref.tmp.sroa.9.0..sroa_idx12 = getelementptr inbounds %struct.St4x8, %struct.St4x8* %ret, i64 0, i32 0, i64 6
  store i32 %.fca.6.extract, i32* %ref.tmp.sroa.9.0..sroa_idx12, align 4
  %ref.tmp.sroa.10.0..sroa_idx14 = getelementptr inbounds %struct.St4x8, %struct.St4x8* %ret, i64 0, i32 0, i64 7
  store i32 %.fca.7.extract, i32* %ref.tmp.sroa.10.0..sroa_idx14, align 4
  ret void
}


define internal fastcc [8 x i32] @callee_St4x8(%struct.St4x8* nocapture noundef readonly byval(%struct.St4x8) align 4 %in) {
  ; CHECK:       .func  (.param .align 16 .b8 func_retval0[32])
  ; CHECK-LABEL: callee_St4x8(
  ; CHECK-NEXT:  .param .align 16 .b8 callee_St4x8_param_0[32]
  ; CHECK:       ld.param.v4.u32 {[[R1:%r[0-9]+]], [[R2:%r[0-9]+]], [[R3:%r[0-9]+]], [[R4:%r[0-9]+]]}, [callee_St4x8_param_0];
  ; CHECK:       ld.param.v4.u32 {[[R5:%r[0-9]+]], [[R6:%r[0-9]+]], [[R7:%r[0-9]+]], [[R8:%r[0-9]+]]}, [callee_St4x8_param_0+16];
  ; CHECK:       st.param.v4.b32 [func_retval0+0],  {[[R1]], [[R2]], [[R3]], [[R4]]};
  ; CHECK:       st.param.v4.b32 [func_retval0+16], {[[R5]], [[R6]], [[R7]], [[R8]]};
  ; CHECK-NEXT:  ret;
  %arrayidx = getelementptr inbounds %struct.St4x8, %struct.St4x8* %in, i64 0, i32 0, i64 0
  %1 = load i32, i32* %arrayidx, align 4
  %arrayidx.1 = getelementptr inbounds %struct.St4x8, %struct.St4x8* %in, i64 0, i32 0, i64 1
  %2 = load i32, i32* %arrayidx.1, align 4
  %arrayidx.2 = getelementptr inbounds %struct.St4x8, %struct.St4x8* %in, i64 0, i32 0, i64 2
  %3 = load i32, i32* %arrayidx.2, align 4
  %arrayidx.3 = getelementptr inbounds %struct.St4x8, %struct.St4x8* %in, i64 0, i32 0, i64 3
  %4 = load i32, i32* %arrayidx.3, align 4
  %arrayidx.4 = getelementptr inbounds %struct.St4x8, %struct.St4x8* %in, i64 0, i32 0, i64 4
  %5 = load i32, i32* %arrayidx.4, align 4
  %arrayidx.5 = getelementptr inbounds %struct.St4x8, %struct.St4x8* %in, i64 0, i32 0, i64 5
  %6 = load i32, i32* %arrayidx.5, align 4
  %arrayidx.6 = getelementptr inbounds %struct.St4x8, %struct.St4x8* %in, i64 0, i32 0, i64 6
  %7 = load i32, i32* %arrayidx.6, align 4
  %arrayidx.7 = getelementptr inbounds %struct.St4x8, %struct.St4x8* %in, i64 0, i32 0, i64 7
  %8 = load i32, i32* %arrayidx.7, align 4
  %9 = insertvalue [8 x i32] poison, i32 %1, 0
  %10 = insertvalue [8 x i32] %9, i32 %2, 1
  %11 = insertvalue [8 x i32] %10, i32 %3, 2
  %12 = insertvalue [8 x i32] %11, i32 %4, 3
  %13 = insertvalue [8 x i32] %12, i32 %5, 4
  %14 = insertvalue [8 x i32] %13, i32 %6, 5
  %15 = insertvalue [8 x i32] %14, i32 %7, 6
  %oldret = insertvalue [8 x i32] %15, i32 %8, 7
  ret [8 x i32] %oldret
}


define dso_local void @caller_St8x1(%struct.St8x1* nocapture noundef readonly byval(%struct.St8x1) align 8 %in, %struct.St8x1* nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func caller_St8x1(
  ; CHECK:               .param .align 8 .b8 caller_St8x1_param_0[8],
  ; CHECK:               .param .b32 caller_St8x1_param_1
  ; CHECK:       )
  ; CHECK:       .param .b64 param0;
  ; CHECK:       st.param.b64 [param0+0], {{%rd[0-9]+}};
  ; CHECK:       .param .align 16 .b8 retval0[8];
  ; CHECK:       call.uni (retval0),
  ; CHECK-NEXT:  callee_St8x1,
  ; CHECK-NEXT:  (
  ; CHECK-NEXT:  param0
  ; CHECK-NEXT:  );
  ; CHECK:       ld.param.b64 {{%rd[0-9]+}}, [retval0+0];
  %1 = getelementptr inbounds %struct.St8x1, %struct.St8x1* %in, i64 0, i32 0, i64 0
  %2 = load i64, i64* %1, align 8
  %call = tail call fastcc [1 x i64] @callee_St8x1(i64 %2)
  %.fca.0.extract = extractvalue [1 x i64] %call, 0
  %ref.tmp.sroa.0.0..sroa_idx = getelementptr inbounds %struct.St8x1, %struct.St8x1* %ret, i64 0, i32 0, i64 0
  store i64 %.fca.0.extract, i64* %ref.tmp.sroa.0.0..sroa_idx, align 8
  ret void
}


define internal fastcc [1 x i64] @callee_St8x1(i64 %in.0.val) {
  ; CHECK:       .func  (.param .align 16 .b8 func_retval0[8])
  ; CHECK-LABEL: callee_St8x1(
  ; CHECK-NEXT:  .param .b64 callee_St8x1_param_0
  ; CHECK:       ld.param.u64 [[RD1:%rd[0-9]+]], [callee_St8x1_param_0];
  ; CHECK:       st.param.b64 [func_retval0+0],  [[RD1]];
  ; CHECK-NEXT:  ret;
  %oldret = insertvalue [1 x i64] poison, i64 %in.0.val, 0
  ret [1 x i64] %oldret
}


define dso_local void @caller_St8x2(%struct.St8x2* nocapture noundef readonly byval(%struct.St8x2) align 8 %in, %struct.St8x2* nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func caller_St8x2(
  ; CHECK:               .param .align 8 .b8 caller_St8x2_param_0[16],
  ; CHECK:               .param .b32 caller_St8x2_param_1
  ; CHECK:       )
  ; CHECK:       .param .align 16 .b8 param0[16];
  ; CHECK:       st.param.v2.b64 [param0+0],  {{{%rd[0-9]+}}, {{%rd[0-9]+}}};
  ; CHECK:       .param .align 16 .b8 retval0[16];
  ; CHECK:       call.uni (retval0),
  ; CHECK-NEXT:  callee_St8x2,
  ; CHECK-NEXT:  (
  ; CHECK-NEXT:  param0
  ; CHECK-NEXT:  );
  ; CHECK:       ld.param.v2.b64 {{{%rd[0-9]+}}, {{%rd[0-9]+}}}, [retval0+0];
  %call = tail call fastcc [2 x i64] @callee_St8x2(%struct.St8x2* noundef nonnull byval(%struct.St8x2) align 8 %in)
  %.fca.0.extract = extractvalue [2 x i64] %call, 0
  %.fca.1.extract = extractvalue [2 x i64] %call, 1
  %ref.tmp.sroa.0.0..sroa_idx = getelementptr inbounds %struct.St8x2, %struct.St8x2* %ret, i64 0, i32 0, i64 0
  store i64 %.fca.0.extract, i64* %ref.tmp.sroa.0.0..sroa_idx, align 8
  %ref.tmp.sroa.4.0..sroa_idx3 = getelementptr inbounds %struct.St8x2, %struct.St8x2* %ret, i64 0, i32 0, i64 1
  store i64 %.fca.1.extract, i64* %ref.tmp.sroa.4.0..sroa_idx3, align 8
  ret void
}


define internal fastcc [2 x i64] @callee_St8x2(%struct.St8x2* nocapture noundef readonly byval(%struct.St8x2) align 8 %in) {
  ; CHECK:       .func  (.param .align 16 .b8 func_retval0[16])
  ; CHECK-LABEL: callee_St8x2(
  ; CHECK-NEXT:  .param .align 16 .b8 callee_St8x2_param_0[16]
  ; CHECK:       ld.param.v2.u64 {[[RD1:%rd[0-9]+]], [[RD2:%rd[0-9]+]]}, [callee_St8x2_param_0];
  ; CHECK:       st.param.v2.b64 [func_retval0+0], {[[RD1]], [[RD2]]};
  ; CHECK-NEXT:  ret;
  %arrayidx = getelementptr inbounds %struct.St8x2, %struct.St8x2* %in, i64 0, i32 0, i64 0
  %1 = load i64, i64* %arrayidx, align 8
  %arrayidx.1 = getelementptr inbounds %struct.St8x2, %struct.St8x2* %in, i64 0, i32 0, i64 1
  %2 = load i64, i64* %arrayidx.1, align 8
  %3 = insertvalue [2 x i64] poison, i64 %1, 0
  %oldret = insertvalue [2 x i64] %3, i64 %2, 1
  ret [2 x i64] %oldret
}


define dso_local void @caller_St8x3(%struct.St8x3* nocapture noundef readonly byval(%struct.St8x3) align 8 %in, %struct.St8x3* nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func caller_St8x3(
  ; CHECK:               .param .align 8 .b8 caller_St8x3_param_0[24],
  ; CHECK:               .param .b32 caller_St8x3_param_1
  ; CHECK:       )
  ; CHECK:       .param .align 16 .b8 param0[24];
  ; CHECK:       st.param.v2.b64 [param0+0],  {{{%rd[0-9]+}}, {{%rd[0-9]+}}};
  ; CHECK:       st.param.b64    [param0+16], {{%rd[0-9]+}};
  ; CHECK:       .param .align 16 .b8 retval0[24];
  ; CHECK:       call.uni (retval0),
  ; CHECK-NEXT:  callee_St8x3,
  ; CHECK-NEXT:  (
  ; CHECK-NEXT:  param0
  ; CHECK-NEXT:  );
  ; CHECK:       ld.param.v2.b64 {{{%rd[0-9]+}}, {{%rd[0-9]+}}}, [retval0+0];
  ; CHECK:       ld.param.b64    {{%rd[0-9]+}}, [retval0+16];
  %call = tail call fastcc [3 x i64] @callee_St8x3(%struct.St8x3* noundef nonnull byval(%struct.St8x3) align 8 %in)
  %.fca.0.extract = extractvalue [3 x i64] %call, 0
  %.fca.1.extract = extractvalue [3 x i64] %call, 1
  %.fca.2.extract = extractvalue [3 x i64] %call, 2
  %ref.tmp.sroa.0.0..sroa_idx = getelementptr inbounds %struct.St8x3, %struct.St8x3* %ret, i64 0, i32 0, i64 0
  store i64 %.fca.0.extract, i64* %ref.tmp.sroa.0.0..sroa_idx, align 8
  %ref.tmp.sroa.4.0..sroa_idx2 = getelementptr inbounds %struct.St8x3, %struct.St8x3* %ret, i64 0, i32 0, i64 1
  store i64 %.fca.1.extract, i64* %ref.tmp.sroa.4.0..sroa_idx2, align 8
  %ref.tmp.sroa.5.0..sroa_idx4 = getelementptr inbounds %struct.St8x3, %struct.St8x3* %ret, i64 0, i32 0, i64 2
  store i64 %.fca.2.extract, i64* %ref.tmp.sroa.5.0..sroa_idx4, align 8
  ret void
}


define internal fastcc [3 x i64] @callee_St8x3(%struct.St8x3* nocapture noundef readonly byval(%struct.St8x3) align 8 %in) {
  ; CHECK:       .func  (.param .align 16 .b8 func_retval0[24])
  ; CHECK-LABEL: callee_St8x3(
  ; CHECK-NEXT:  .param .align 16 .b8 callee_St8x3_param_0[24]
  ; CHECK:       ld.param.v2.u64 {[[RD1:%rd[0-9]+]], [[RD2:%rd[0-9]+]]}, [callee_St8x3_param_0];
  ; CHECK:       ld.param.u64    [[RD3:%rd[0-9]+]],  [callee_St8x3_param_0+16];
  ; CHECK:       st.param.v2.b64 [func_retval0+0],   {[[RD1]], [[RD2]]};
  ; CHECK:       st.param.b64    [func_retval0+16],  [[RD3]];
  ; CHECK-NEXT:  ret;
  %arrayidx = getelementptr inbounds %struct.St8x3, %struct.St8x3* %in, i64 0, i32 0, i64 0
  %1 = load i64, i64* %arrayidx, align 8
  %arrayidx.1 = getelementptr inbounds %struct.St8x3, %struct.St8x3* %in, i64 0, i32 0, i64 1
  %2 = load i64, i64* %arrayidx.1, align 8
  %arrayidx.2 = getelementptr inbounds %struct.St8x3, %struct.St8x3* %in, i64 0, i32 0, i64 2
  %3 = load i64, i64* %arrayidx.2, align 8
  %4 = insertvalue [3 x i64] poison, i64 %1, 0
  %5 = insertvalue [3 x i64] %4, i64 %2, 1
  %oldret = insertvalue [3 x i64] %5, i64 %3, 2
  ret [3 x i64] %oldret
}


define dso_local void @caller_St8x4(%struct.St8x4* nocapture noundef readonly byval(%struct.St8x4) align 8 %in, %struct.St8x4* nocapture noundef writeonly %ret) {
  ; CHECK-LABEL: .visible .func caller_St8x4(
  ; CHECK:               .param .align 8 .b8 caller_St8x4_param_0[32],
  ; CHECK:               .param .b32 caller_St8x4_param_1
  ; CHECK:       )
  ; CHECK:       .param .align 16 .b8 param0[32];
  ; CHECK:       st.param.v2.b64 [param0+0],  {{{%rd[0-9]+}}, {{%rd[0-9]+}}};
  ; CHECK:       st.param.v2.b64 [param0+16], {{{%rd[0-9]+}}, {{%rd[0-9]+}}};
  ; CHECK:       .param .align 16 .b8 retval0[32];
  ; CHECK:       call.uni (retval0),
  ; CHECK-NEXT:  callee_St8x4,
  ; CHECK-NEXT:  (
  ; CHECK-NEXT:  param0
  ; CHECK-NEXT:  );
  ; CHECK:       ld.param.v2.b64 {{{%rd[0-9]+}}, {{%rd[0-9]+}}}, [retval0+0];
  ; CHECK:       ld.param.v2.b64 {{{%rd[0-9]+}}, {{%rd[0-9]+}}}, [retval0+16];
  %call = tail call fastcc [4 x i64] @callee_St8x4(%struct.St8x4* noundef nonnull byval(%struct.St8x4) align 8 %in)
  %.fca.0.extract = extractvalue [4 x i64] %call, 0
  %.fca.1.extract = extractvalue [4 x i64] %call, 1
  %.fca.2.extract = extractvalue [4 x i64] %call, 2
  %.fca.3.extract = extractvalue [4 x i64] %call, 3
  %ref.tmp.sroa.0.0..sroa_idx = getelementptr inbounds %struct.St8x4, %struct.St8x4* %ret, i64 0, i32 0, i64 0
  store i64 %.fca.0.extract, i64* %ref.tmp.sroa.0.0..sroa_idx, align 8
  %ref.tmp.sroa.4.0..sroa_idx3 = getelementptr inbounds %struct.St8x4, %struct.St8x4* %ret, i64 0, i32 0, i64 1
  store i64 %.fca.1.extract, i64* %ref.tmp.sroa.4.0..sroa_idx3, align 8
  %ref.tmp.sroa.5.0..sroa_idx5 = getelementptr inbounds %struct.St8x4, %struct.St8x4* %ret, i64 0, i32 0, i64 2
  store i64 %.fca.2.extract, i64* %ref.tmp.sroa.5.0..sroa_idx5, align 8
  %ref.tmp.sroa.6.0..sroa_idx7 = getelementptr inbounds %struct.St8x4, %struct.St8x4* %ret, i64 0, i32 0, i64 3
  store i64 %.fca.3.extract, i64* %ref.tmp.sroa.6.0..sroa_idx7, align 8
  ret void
}


define internal fastcc [4 x i64] @callee_St8x4(%struct.St8x4* nocapture noundef readonly byval(%struct.St8x4) align 8 %in) {
  ; CHECK:       .func  (.param .align 16 .b8 func_retval0[32])
  ; CHECK-LABEL: callee_St8x4(
  ; CHECK-NEXT:  .param .align 16 .b8 callee_St8x4_param_0[32]
  ; CHECK:       ld.param.v2.u64 {[[RD1:%rd[0-9]+]], [[RD2:%rd[0-9]+]]}, [callee_St8x4_param_0];
  ; CHECK:       ld.param.v2.u64 {[[RD3:%rd[0-9]+]], [[RD4:%rd[0-9]+]]}, [callee_St8x4_param_0+16];
  ; CHECK:       st.param.v2.b64 [func_retval0+0],  {[[RD1]], [[RD2]]};
  ; CHECK:       st.param.v2.b64 [func_retval0+16], {[[RD3]], [[RD4]]};
  ; CHECK-NEXT:  ret;
  %arrayidx = getelementptr inbounds %struct.St8x4, %struct.St8x4* %in, i64 0, i32 0, i64 0
  %1 = load i64, i64* %arrayidx, align 8
  %arrayidx.1 = getelementptr inbounds %struct.St8x4, %struct.St8x4* %in, i64 0, i32 0, i64 1
  %2 = load i64, i64* %arrayidx.1, align 8
  %arrayidx.2 = getelementptr inbounds %struct.St8x4, %struct.St8x4* %in, i64 0, i32 0, i64 2
  %3 = load i64, i64* %arrayidx.2, align 8
  %arrayidx.3 = getelementptr inbounds %struct.St8x4, %struct.St8x4* %in, i64 0, i32 0, i64 3
  %4 = load i64, i64* %arrayidx.3, align 8
  %5 = insertvalue [4 x i64] poison, i64 %1, 0
  %6 = insertvalue [4 x i64] %5, i64 %2, 1
  %7 = insertvalue [4 x i64] %6, i64 %3, 2
  %oldret = insertvalue [4 x i64] %7, i64 %4, 3
  ret [4 x i64] %oldret
}

; Section 2 - checking that function argument (including retval) vectorization is done with private linkage.

define private fastcc [4 x i32] @callee_St4x4_private(%struct.St4x4* nocapture noundef readonly byval(%struct.St4x4) align 4 %in) {
  ; CHECK:       .func  (.param .align 16 .b8 func_retval0[16])
  ; CHECK-LABEL: callee_St4x4_private(
  ; CHECK-NEXT:  .param .align 16 .b8 callee_St4x4_private_param_0[16]
  ; CHECK:       ld.param.v4.u32 {[[R1:%r[0-9]+]], [[R2:%r[0-9]+]], [[R3:%r[0-9]+]], [[R4:%r[0-9]+]]}, [callee_St4x4_private_param_0];
  ; CHECK:       st.param.v4.b32 [func_retval0+0], {[[R1]], [[R2]], [[R3]], [[R4]]};
  ; CHECK-NEXT:  ret;
  %arrayidx = getelementptr inbounds %struct.St4x4, %struct.St4x4* %in, i64 0, i32 0, i64 0
  %1 = load i32, i32* %arrayidx, align 4
  %arrayidx.1 = getelementptr inbounds %struct.St4x4, %struct.St4x4* %in, i64 0, i32 0, i64 1
  %2 = load i32, i32* %arrayidx.1, align 4
  %arrayidx.2 = getelementptr inbounds %struct.St4x4, %struct.St4x4* %in, i64 0, i32 0, i64 2
  %3 = load i32, i32* %arrayidx.2, align 4
  %arrayidx.3 = getelementptr inbounds %struct.St4x4, %struct.St4x4* %in, i64 0, i32 0, i64 3
  %4 = load i32, i32* %arrayidx.3, align 4
  %5 = insertvalue [4 x i32] poison, i32 %1, 0
  %6 = insertvalue [4 x i32] %5, i32 %2, 1
  %7 = insertvalue [4 x i32] %6, i32 %3, 2
  %oldret = insertvalue [4 x i32] %7, i32 %4, 3
  ret [4 x i32] %oldret
}

; Section 3 - checking that function argument (including retval) vectorization
; is NOT done with linkage types other than internal and private.

define external fastcc [4 x i32] @callee_St4x4_external(%struct.St4x4* nocapture noundef readonly byval(%struct.St4x4) align 4 %in) {
  ; CHECK:       .func  (.param .align 4 .b8 func_retval0[16])
  ; CHECK-LABEL: callee_St4x4_external(
  ; CHECK-NEXT:  .param .align 4 .b8 callee_St4x4_external_param_0[16]
  ; CHECK:       ld.param.u32 [[R1:%r[0-9]+]],   [callee_St4x4_external_param_0];
  ; CHECK:       ld.param.u32 [[R2:%r[0-9]+]],   [callee_St4x4_external_param_0+4];
  ; CHECK:       ld.param.u32 [[R3:%r[0-9]+]],   [callee_St4x4_external_param_0+8];
  ; CHECK:       ld.param.u32 [[R4:%r[0-9]+]],   [callee_St4x4_external_param_0+12];
  ; CHECK:       st.param.b32 [func_retval0+0],  [[R1]];
  ; CHECK:       st.param.b32 [func_retval0+4],  [[R2]];
  ; CHECK:       st.param.b32 [func_retval0+8],  [[R3]];
  ; CHECK:       st.param.b32 [func_retval0+12], [[R4]];
  ; CHECK-NEXT:  ret;
  %arrayidx = getelementptr inbounds %struct.St4x4, %struct.St4x4* %in, i64 0, i32 0, i64 0
  %1 = load i32, i32* %arrayidx, align 4
  %arrayidx.1 = getelementptr inbounds %struct.St4x4, %struct.St4x4* %in, i64 0, i32 0, i64 1
  %2 = load i32, i32* %arrayidx.1, align 4
  %arrayidx.2 = getelementptr inbounds %struct.St4x4, %struct.St4x4* %in, i64 0, i32 0, i64 2
  %3 = load i32, i32* %arrayidx.2, align 4
  %arrayidx.3 = getelementptr inbounds %struct.St4x4, %struct.St4x4* %in, i64 0, i32 0, i64 3
  %4 = load i32, i32* %arrayidx.3, align 4
  %5 = insertvalue [4 x i32] poison, i32 %1, 0
  %6 = insertvalue [4 x i32] %5, i32 %2, 1
  %7 = insertvalue [4 x i32] %6, i32 %3, 2
  %oldret = insertvalue [4 x i32] %7, i32 %4, 3
  ret [4 x i32] %oldret
}
