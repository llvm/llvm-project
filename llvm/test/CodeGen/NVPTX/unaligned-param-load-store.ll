; Verifies correctness of load/store of parameters and return values.
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_35 -O0 -verify-machineinstrs | FileCheck -allow-deprecated-dag-overlap %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_35 -O0 -verify-machineinstrs | %ptxas-verify %}

%s_i8i16p = type { <{ i16, i8, i16 }>, i64 }
%s_i8i32p = type { <{ i32, i8, i32 }>, i64 }
%s_i8i64p = type { <{ i64, i8, i64 }>, i64 }
%s_i8f16p = type { <{ half, i8, half }>, i64 }
%s_i8f16x2p = type { <{ <2 x half>, i8, <2 x half> }>, i64 }
%s_i8f32p = type { <{ float, i8, float }>, i64 }
%s_i8f64p = type { <{ double, i8, double }>, i64 }

; -- All loads/stores from parameters aligned by one must be done one
;    byte at a time.
; -- Notes:
;   -- There are two fields of interest in the packed part of the struct, one
;      with a proper offset and one without. The former should be loaded or
;      stored as a whole, and the latter by bytes.
;   -- Only loading and storing the said fields are checked in the following
;      series of tests so that they are more concise.

; CHECK:       .visible .func (.param .align 8 .b8 func_retval0[16])
; CHECK-LABEL: test_s_i8i16p(
; CHECK:        .param .align 8 .b8 test_s_i8i16p_param_0[16]
; CHECK-DAG:    ld.param.b16 [[P0:%rs[0-9]+]],   [test_s_i8i16p_param_0];
; CHECK-DAG:    ld.param.b8 [[P2_0:%rs[0-9]+]],   [test_s_i8i16p_param_0+3];
; CHECK-DAG:    ld.param.b8 [[P2_1:%rs[0-9]+]],   [test_s_i8i16p_param_0+4];
; CHECK-DAG:    shl.b16     [[P2_1_shl:%rs[0-9]+]], [[P2_1]], 8;
; CHECK-DAG:    or.b16      [[P2_1_or:%rs[0-9]+]], [[P2_1_shl]], [[P2_0]];
; CHECK:        { // callseq
; CHECK:        .param .align 8 .b8 param0[16];
; CHECK-DAG:    st.param.b16 [param0], [[P0]];
; CHECK-DAG:    st.param.b8  [param0+3], [[P2_1_or]];
; CHECK-DAG:    st.param.b8  [param0+4], [[P2_1]];
; CHECK:        .param .align 8 .b8 retval0[16];
; CHECK-NEXT:   call.uni (retval0),
; CHECK-NEXT:   test_s_i8i16p,
; CHECK-NEXT:   (
; CHECK-NEXT:   param0
; CHECK-NEXT:   );
; CHECK-DAG:    ld.param.b16 [[R0:%rs[0-9]+]],   [retval0];
; CHECK-DAG:    ld.param.b8  [[R2_0:%rs[0-9]+]], [retval0+3];
; CHECK-DAG:    ld.param.b8  [[R2_1:%rs[0-9]+]], [retval0+4];
; CHECK:        } // callseq
; CHECK-DAG:    st.param.b16 [func_retval0], [[R0]];
; CHECK-DAG:    shl.b16      [[R2_1_shl:%rs[0-9]+]], [[R2_1]], 8;
; CHECK-DAG:    and.b16      [[R2_0_and:%rs[0-9]+]], [[R2_0]], 255;
; CHECK-DAG:    or.b16       [[R2:%rs[0-9]+]], [[R2_0_and]], [[R2_1_shl]];
; CHECK-DAG:    st.param.b8  [func_retval0+3], [[R2]];
; CHECK-DAG:    and.b16      [[R2_1_and:%rs[0-9]+]], [[R2_1]], 255;
; CHECK-DAG:    st.param.b8  [func_retval0+4], [[R2_1_and]];
; CHECK:        ret;

define %s_i8i16p @test_s_i8i16p(%s_i8i16p %a) {
       %r = tail call %s_i8i16p @test_s_i8i16p(%s_i8i16p %a)
       ret %s_i8i16p %r
}

; CHECK:       .visible .func (.param .align 8 .b8 func_retval0[24])
; CHECK-LABEL: test_s_i8i32p(
; CHECK:        .param .align 8 .b8 test_s_i8i32p_param_0[24]
; CHECK-DAG:    ld.param.b32 [[P0:%r[0-9]+]],   [test_s_i8i32p_param_0];
; CHECK-DAG:    ld.param.b8 [[P2_0:%r[0-9]+]],   [test_s_i8i32p_param_0+5];
; CHECK-DAG:    ld.param.b8 [[P2_1:%r[0-9]+]],   [test_s_i8i32p_param_0+6];
; CHECK-DAG:    ld.param.b8 [[P2_2:%r[0-9]+]],   [test_s_i8i32p_param_0+7];
; CHECK-DAG:    ld.param.b8 [[P2_3:%r[0-9]+]],   [test_s_i8i32p_param_0+8];
; CHECK-DAG:    shl.b32     [[P2_1_shl:%r[0-9]+]], [[P2_1]], 8;
; CHECK-DAG:    shl.b32     [[P2_2_shl:%r[0-9]+]], [[P2_2]], 16;
; CHECK-DAG:    shl.b32     [[P2_3_shl:%r[0-9]+]], [[P2_3]], 24;
; CHECK-DAG:    or.b32      [[P2_or:%r[0-9]+]], [[P2_1_shl]], [[P2_0]];
; CHECK-DAG:    or.b32      [[P2_or_1:%r[0-9]+]], [[P2_3_shl]], [[P2_2_shl]];
; CHECK-DAG:    or.b32      [[P2:%r[0-9]+]], [[P2_or_1]], [[P2_or]];
; CHECK-DAG:    shr.u32     [[P2_1_shr:%r[0-9]+]], [[P2]], 8;
; CHECK-DAG:    shr.u32     [[P2_2_shr:%r[0-9]+]], [[P2_or_1]], 16;
; CHECK:        { // callseq
; CHECK-DAG:    .param .align 8 .b8 param0[24];
; CHECK-DAG:    st.param.b32 [param0], [[P0]];
; CHECK-DAG:    st.param.b8  [param0+5], [[P2]];
; CHECK-DAG:    st.param.b8  [param0+6], [[P2_1_shr]];
; CHECK-DAG:    st.param.b8  [param0+7], [[P2_2_shr]];
; CHECK-DAG:    st.param.b8  [param0+8], [[P2_3]];
; CHECK:        .param .align 8 .b8 retval0[24];
; CHECK-NEXT:   call.uni (retval0),
; CHECK-NEXT:   test_s_i8i32p,
; CHECK-NEXT:   (
; CHECK-NEXT:   param0
; CHECK-NEXT:   );
; CHECK-DAG:    ld.param.b32 [[R0:%r[0-9]+]],   [retval0];
; CHECK-DAG:    ld.param.b8  [[R2_0:%rs[0-9]+]], [retval0+5];
; CHECK-DAG:    ld.param.b8  [[R2_1:%rs[0-9]+]], [retval0+6];
; CHECK-DAG:    ld.param.b8  [[R2_2:%rs[0-9]+]], [retval0+7];
; CHECK-DAG:    ld.param.b8  [[R2_3:%rs[0-9]+]], [retval0+8];
; CHECK:        } // callseq
; CHECK-DAG:    st.param.b32 [func_retval0], [[R0]];
; CHECK-DAG:    st.param.b8  [func_retval0+5],
; CHECK-DAG:    st.param.b8  [func_retval0+6],
; CHECK-DAG:    st.param.b8  [func_retval0+7],
; CHECK-DAG:    st.param.b8  [func_retval0+8],
; CHECK:        ret;

define %s_i8i32p @test_s_i8i32p(%s_i8i32p %a) {
       %r = tail call %s_i8i32p @test_s_i8i32p(%s_i8i32p %a)
       ret %s_i8i32p %r
}

; CHECK:       .visible .func (.param .align 8 .b8 func_retval0[32])
; CHECK-LABEL: test_s_i8i64p(
; CHECK:        .param .align 8 .b8 test_s_i8i64p_param_0[32]
; CHECK-DAG:    ld.param.b64 [[P0:%rd[0-9]+]],   [test_s_i8i64p_param_0];
; CHECK-DAG:    ld.param.b8 [[P2_0:%rd[0-9]+]],   [test_s_i8i64p_param_0+9];
; CHECK-DAG:    ld.param.b8 [[P2_1:%rd[0-9]+]],   [test_s_i8i64p_param_0+10];
; CHECK-DAG:    ld.param.b8 [[P2_2:%rd[0-9]+]],   [test_s_i8i64p_param_0+11];
; CHECK-DAG:    ld.param.b8 [[P2_3:%rd[0-9]+]],   [test_s_i8i64p_param_0+12];
; CHECK-DAG:    ld.param.b8 [[P2_4:%rd[0-9]+]],   [test_s_i8i64p_param_0+13];
; CHECK-DAG:    ld.param.b8 [[P2_5:%rd[0-9]+]],   [test_s_i8i64p_param_0+14];
; CHECK-DAG:    ld.param.b8 [[P2_6:%rd[0-9]+]],   [test_s_i8i64p_param_0+15];
; CHECK-DAG:    ld.param.b8 [[P2_7:%rd[0-9]+]],   [test_s_i8i64p_param_0+16];
; CHECK-DAG:    shl.b64      [[P2_1_shl:%rd[0-9]+]], [[P2_1]], 8;
; CHECK-DAG:    shl.b64      [[P2_2_shl:%rd[0-9]+]], [[P2_2]], 16;
; CHECK-DAG:    shl.b64      [[P2_3_shl:%rd[0-9]+]], [[P2_3]], 24;
; CHECK-DAG:    or.b64       [[P2_or_0:%rd[0-9]+]], [[P2_1_shl]], [[P2_0]];
; CHECK-DAG:    or.b64       [[P2_or_1:%rd[0-9]+]], [[P2_3_shl]], [[P2_2_shl]];
; CHECK-DAG:    or.b64       [[P2_or_2:%rd[0-9]+]], [[P2_or_1]], [[P2_or_0]];
; CHECK-DAG:    shl.b64 	 [[P2_5_shl:%rd[0-9]+]], [[P2_5]], 8;
; CHECK-DAG:    shl.b64      [[P2_6_shl:%rd[0-9]+]], [[P2_6]], 16;
; CHECK-DAG:    shl.b64      [[P2_7_shl:%rd[0-9]+]], [[P2_7]], 24;
; CHECK-DAG:    or.b64       [[P2_or_3:%rd[0-9]+]], [[P2_5_shl]], [[P2_4]];
; CHECK-DAG:    or.b64       [[P2_or_4:%rd[0-9]+]], [[P2_7_shl]], [[P2_6_shl]];
; CHECK-DAG:    or.b64       [[P2_or_5:%rd[0-9]+]], [[P2_or_4]], [[P2_or_3]];
; CHECK-DAG:    shl.b64      [[P2_or_shl:%rd[0-9]+]], [[P2_or_5]], 32;
; CHECK-DAG:    or.b64       [[P2:%rd[0-9]+]], [[P2_or_shl]], [[P2_or_2]];
; CHECK-DAG:    shr.u64      [[P2_shr_1:%rd[0-9]+]], [[P2]], 8;
; CHECK-DAG:    shr.u64      [[P2_shr_2:%rd[0-9]+]], [[P2]], 16;
; CHECK-DAG:    shr.u64      [[P2_shr_3:%rd[0-9]+]], [[P2]], 24;
; CHECK-DAG:    bfe.u64      [[P2_bfe_4:%rd[0-9]+]], [[P2_or_5]], 8, 24;
; CHECK-DAG:    bfe.u64      [[P2_bfe_5:%rd[0-9]+]], [[P2_or_5]], 16, 16;
; CHECK-DAG:    bfe.u64      [[P2_bfe_6:%rd[0-9]+]], [[P2_or_5]], 24, 8;
; CHECK:        { // callseq
; CHECK:        .param .align 8 .b8 param0[32];
; CHECK-DAG:    st.param.b64 [param0],  [[P0]];
; CHECK-DAG:    st.param.b8  [param0+9],  [[P2]];
; CHECK-DAG:    st.param.b8  [param0+10], [[P2_shr_1]];
; CHECK-DAG:    st.param.b8  [param0+11], [[P2_shr_2]];
; CHECK-DAG:    st.param.b8  [param0+12], [[P2_shr_3]];
; CHECK-DAG:    st.param.b8  [param0+13], [[P2_or_5]];
; CHECK-DAG:    st.param.b8  [param0+14], [[P2_bfe_4]];
; CHECK-DAG:    st.param.b8  [param0+15], [[P2_bfe_5]];
; CHECK-DAG:    st.param.b8  [param0+16], [[P2_bfe_6]];
; CHECK:        .param .align 8 .b8 retval0[32];
; CHECK-NEXT:   call.uni (retval0),
; CHECK-NEXT:   test_s_i8i64p,
; CHECK-NEXT:   (
; CHECK-NEXT:   param0
; CHECK-NEXT:   );
; CHECK-DAG:    ld.param.b64 [[R0:%rd[0-9]+]],   [retval0];
; CHECK-DAG:    ld.param.b8  [[R2_0:%rs[0-9]+]], [retval0+9];
; CHECK-DAG:    ld.param.b8  [[R2_1:%rs[0-9]+]], [retval0+10];
; CHECK-DAG:    ld.param.b8  [[R2_2:%rs[0-9]+]], [retval0+11];
; CHECK-DAG:    ld.param.b8  [[R2_3:%rs[0-9]+]], [retval0+12];
; CHECK-DAG:    ld.param.b8  [[R2_4:%rs[0-9]+]], [retval0+13];
; CHECK-DAG:    ld.param.b8  [[R2_5:%rs[0-9]+]], [retval0+14];
; CHECK-DAG:    ld.param.b8  [[R2_6:%rs[0-9]+]], [retval0+15];
; CHECK-DAG:    ld.param.b8  [[R2_7:%rs[0-9]+]], [retval0+16];
; CHECK:        } // callseq
; CHECK-DAG:    st.param.b64 [func_retval0], [[R0]];
; CHECK-DAG:    st.param.b8  [func_retval0+9],
; CHECK-DAG:    st.param.b8  [func_retval0+10],
; CHECK-DAG:    st.param.b8  [func_retval0+11],
; CHECK-DAG:    st.param.b8  [func_retval0+12],
; CHECK-DAG:    st.param.b8  [func_retval0+13],
; CHECK-DAG:    st.param.b8  [func_retval0+14],
; CHECK-DAG:    st.param.b8  [func_retval0+15],
; CHECK-DAG:    st.param.b8  [func_retval0+16],
; CHECK:        ret;

define %s_i8i64p @test_s_i8i64p(%s_i8i64p %a) {
       %r = tail call %s_i8i64p @test_s_i8i64p(%s_i8i64p %a)
       ret %s_i8i64p %r
}

; CHECK:       .visible .func (.param .align 8 .b8 func_retval0[16])
; CHECK-LABEL: test_s_i8f16p(
; CHECK:        .param .align 8 .b8 test_s_i8f16p_param_0[16]
; CHECK-DAG:    ld.param.b16 [[P0:%rs[0-9]+]],     [test_s_i8f16p_param_0];
; CHECK-DAG:    ld.param.b8  [[P2_0:%rs[0-9]+]],   [test_s_i8f16p_param_0+3];
; CHECK-DAG:    ld.param.b8  [[P2_1:%rs[0-9]+]],   [test_s_i8f16p_param_0+4];
; CHECK-DAG:    shl.b16      [[P2_1_shl:%rs[0-9]+]], [[P2_1]], 8;
; CHECK-DAG:    or.b16       [[P2_1_or:%rs[0-9]+]], [[P2_1_shl]], [[P2_0]];
; CHECK:        { // callseq
; CHECK:        .param .align 8 .b8 param0[16];
; CHECK-DAG:    st.param.b16 [param0], [[P0]];
; CHECK-DAG:    st.param.b8  [param0+3], [[P2_1_or]];
; CHECK-DAG:    st.param.b8  [param0+4], [[P2_1]];
; CHECK:        .param .align 8 .b8 retval0[16];
; CHECK-NEXT:   call.uni (retval0),
; CHECK-NEXT:   test_s_i8f16p,
; CHECK-NEXT:   (
; CHECK-NEXT:   param0
; CHECK-NEXT:   );
; CHECK-DAG:    ld.param.b16 [[R0:%rs[0-9]+]],     [retval0];
; CHECK-DAG:    ld.param.b8  [[R2I_0:%rs[0-9]+]], [retval0+3];
; CHECK-DAG:    ld.param.b8  [[R2I_1:%rs[0-9]+]], [retval0+4];
; CHECK:        } // callseq
; CHECK-DAG:    st.param.b16 [func_retval0], [[R0]];
; CHECK-DAG:    shl.b16      [[R2I_1_shl:%rs[0-9]+]], [[R2I_1]], 8;
; CHECK-DAG:    and.b16      [[R2I_0_and:%rs[0-9]+]], [[R2I_0]], 255;
; CHECK-DAG:    or.b16       [[R2I:%rs[0-9]+]], [[R2I_0_and]], [[R2I_1_shl]];
; CHECK-DAG:    st.param.b8  [func_retval0+3],  [[R2I]];
; CHECK-DAG:    and.b16      [[R2I_1_and:%rs[0-9]+]], [[R2I_1]], 255;
; CHECK-DAG:    st.param.b8  [func_retval0+4],  [[R2I_1_and]];
; CHECK:        ret;

define %s_i8f16p @test_s_i8f16p(%s_i8f16p %a) {
       %r = tail call %s_i8f16p @test_s_i8f16p(%s_i8f16p %a)
       ret %s_i8f16p %r
}

; CHECK:       .visible .func (.param .align 8 .b8 func_retval0[24])
; CHECK-LABEL: test_s_i8f16x2p(
; CHECK:        .param .align 8 .b8 test_s_i8f16x2p_param_0[24]
; CHECK-DAG:    ld.param.b32 [[P0:%r[0-9]+]],  [test_s_i8f16x2p_param_0];
; CHECK-DAG:    ld.param.b8  [[P2_0:%r[0-9]+]],   [test_s_i8f16x2p_param_0+5];
; CHECK-DAG:    ld.param.b8  [[P2_1:%r[0-9]+]],   [test_s_i8f16x2p_param_0+6];
; CHECK-DAG:    ld.param.b8  [[P2_2:%r[0-9]+]],   [test_s_i8f16x2p_param_0+7];
; CHECK-DAG:    ld.param.b8  [[P2_3:%r[0-9]+]],   [test_s_i8f16x2p_param_0+8];
; CHECK-DAG:    shl.b32      [[P2_1_shl:%r[0-9]+]], [[P2_1]], 8;
; CHECK-DAG:    shl.b32      [[P2_2_shl:%r[0-9]+]], [[P2_2]], 16;
; CHECK-DAG:    shl.b32      [[P2_3_shl:%r[0-9]+]], [[P2_3]], 24;
; CHECK-DAG:    or.b32       [[P2_or:%r[0-9]+]], [[P2_1_shl]], [[P2_0]];
; CHECK-DAG:    or.b32       [[P2_or_1:%r[0-9]+]], [[P2_3_shl]], [[P2_2_shl]];
; CHECK-DAG:    or.b32       [[P2:%r[0-9]+]], [[P2_or_1]], [[P2_or]];
; CHECK-DAG:    shr.u32      [[P2_1_shr:%r[0-9]+]], [[P2]], 8;
; CHECK-DAG:    shr.u32      [[P2_2_shr:%r[0-9]+]], [[P2_or_1]], 16;
; CHECK:        { // callseq
; CHECK-DAG:    .param .align 8 .b8 param0[24];
; CHECK-DAG:    st.param.b32 [param0], [[P0]];
; CHECK-DAG:    st.param.b8  [param0+5], [[P2]];
; CHECK-DAG:    st.param.b8  [param0+6], [[P2_1_shr]];
; CHECK-DAG:    st.param.b8  [param0+7], [[P2_2_shr]];
; CHECK-DAG:    st.param.b8  [param0+8], [[P2_3]];
; CHECK:        .param .align 8 .b8 retval0[24];
; CHECK-NEXT:   call.uni (retval0),
; CHECK-NEXT:   test_s_i8f16x2p,
; CHECK-NEXT:   (
; CHECK-NEXT:   param0
; CHECK-NEXT:   );
; CHECK-DAG:    ld.param.b32 [[R0:%r[0-9]+]],   [retval0];
; CHECK-DAG:    ld.param.b8  [[R2_0:%rs[0-9]+]], [retval0+5];
; CHECK-DAG:    ld.param.b8  [[R2_1:%rs[0-9]+]], [retval0+6];
; CHECK-DAG:    ld.param.b8  [[R2_2:%rs[0-9]+]], [retval0+7];
; CHECK-DAG:    ld.param.b8  [[R2_3:%rs[0-9]+]], [retval0+8];
; CHECK:        } // callseq
; CHECK-DAG:    st.param.b32 [func_retval0], [[R0]];
; CHECK-DAG:    st.param.b8  [func_retval0+5],
; CHECK-DAG:    st.param.b8  [func_retval0+6],
; CHECK-DAG:    st.param.b8  [func_retval0+7],
; CHECK-DAG:    st.param.b8  [func_retval0+8],
; CHECK:        ret;

define %s_i8f16x2p @test_s_i8f16x2p(%s_i8f16x2p %a) {
       %r = tail call %s_i8f16x2p @test_s_i8f16x2p(%s_i8f16x2p %a)
       ret %s_i8f16x2p %r
}

; CHECK:       .visible .func (.param .align 8 .b8 func_retval0[24])
; CHECK-LABEL: test_s_i8f32p(
; CHECK:        .param .align 8 .b8 test_s_i8f32p_param_0[24]
; CHECK-DAG:    ld.param.b32 [[P0:%r[0-9]+]],    [test_s_i8f32p_param_0];
; CHECK-DAG:    ld.param.b8  [[P2_0:%r[0-9]+]],   [test_s_i8f32p_param_0+5];
; CHECK-DAG:    ld.param.b8  [[P2_1:%r[0-9]+]],   [test_s_i8f32p_param_0+6];
; CHECK-DAG:    ld.param.b8  [[P2_2:%r[0-9]+]],   [test_s_i8f32p_param_0+7];
; CHECK-DAG:    ld.param.b8  [[P2_3:%r[0-9]+]],   [test_s_i8f32p_param_0+8];
; CHECK-DAG:    shl.b32      [[P2_1_shl:%r[0-9]+]], [[P2_1]], 8;
; CHECK-DAG:    shl.b32      [[P2_2_shl:%r[0-9]+]], [[P2_2]], 16;
; CHECK-DAG:    shl.b32      [[P2_3_shl:%r[0-9]+]], [[P2_3]], 24;
; CHECK-DAG:    or.b32       [[P2_or:%r[0-9]+]], [[P2_1_shl]], [[P2_0]];
; CHECK-DAG:    or.b32       [[P2_or_1:%r[0-9]+]], [[P2_3_shl]], [[P2_2_shl]];
; CHECK-DAG:    or.b32       [[P2:%r[0-9]+]], [[P2_or_1]], [[P2_or]];
; CHECK-DAG:    shr.u32      [[P2_1_shr:%r[0-9]+]], [[P2]], 8;
; CHECK-DAG:    shr.u32      [[P2_2_shr:%r[0-9]+]], [[P2_or_1]], 16;
; CHECK:        { // callseq
; CHECK-DAG:    .param .align 8 .b8 param0[24];
; CHECK-DAG:    st.param.b32 [param0], [[P0]];
; CHECK-DAG:    st.param.b8  [param0+5], [[P2]];
; CHECK-DAG:    st.param.b8  [param0+6], [[P2_1_shr]];
; CHECK-DAG:    st.param.b8  [param0+7], [[P2_2_shr]];
; CHECK-DAG:    st.param.b8  [param0+8], [[P2_3]];
; CHECK:        .param .align 8 .b8 retval0[24];
; CHECK-NEXT:   call.uni (retval0),
; CHECK-NEXT:   test_s_i8f32p,
; CHECK-NEXT:   (
; CHECK-NEXT:   param0
; CHECK-NEXT:   );
; CHECK-DAG:    ld.param.b32 [[R0:%r[0-9]+]],    [retval0];
; CHECK-DAG:    ld.param.b8  [[R2_0:%rs[0-9]+]], [retval0+5];
; CHECK-DAG:    ld.param.b8  [[R2_1:%rs[0-9]+]], [retval0+6];
; CHECK-DAG:    ld.param.b8  [[R2_2:%rs[0-9]+]], [retval0+7];
; CHECK-DAG:    ld.param.b8  [[R2_3:%rs[0-9]+]], [retval0+8];
; CHECK:        } // callseq
; CHECK-DAG:    st.param.b32 [func_retval0], [[R0]];
; CHECK-DAG:    st.param.b8  [func_retval0+5],
; CHECK-DAG:    st.param.b8  [func_retval0+6],
; CHECK-DAG:    st.param.b8  [func_retval0+7],
; CHECK-DAG:    st.param.b8  [func_retval0+8],
; CHECK:        ret;

define %s_i8f32p @test_s_i8f32p(%s_i8f32p %a) {
       %r = tail call %s_i8f32p @test_s_i8f32p(%s_i8f32p %a)
       ret %s_i8f32p %r
}

; CHECK:       .visible .func (.param .align 8 .b8 func_retval0[32])
; CHECK-LABEL: test_s_i8f64p(
; CHECK:        .param .align 8 .b8 test_s_i8f64p_param_0[32]
; CHECK-DAG:    ld.param.b64 [[P0:%rd[0-9]+]],    [test_s_i8f64p_param_0];
; CHECK-DAG:    ld.param.b8  [[P2_0:%rd[0-9]+]],   [test_s_i8f64p_param_0+9];
; CHECK-DAG:    ld.param.b8  [[P2_1:%rd[0-9]+]],   [test_s_i8f64p_param_0+10];
; CHECK-DAG:    ld.param.b8  [[P2_2:%rd[0-9]+]],   [test_s_i8f64p_param_0+11];
; CHECK-DAG:    ld.param.b8  [[P2_3:%rd[0-9]+]],   [test_s_i8f64p_param_0+12];
; CHECK-DAG:    ld.param.b8  [[P2_4:%rd[0-9]+]],   [test_s_i8f64p_param_0+13];
; CHECK-DAG:    ld.param.b8  [[P2_5:%rd[0-9]+]],   [test_s_i8f64p_param_0+14];
; CHECK-DAG:    ld.param.b8  [[P2_6:%rd[0-9]+]],   [test_s_i8f64p_param_0+15];
; CHECK-DAG:    ld.param.b8  [[P2_7:%rd[0-9]+]],   [test_s_i8f64p_param_0+16];
; CHECK-DAG:    shl.b64      [[P2_1_shl:%rd[0-9]+]], [[P2_1]], 8;
; CHECK-DAG:    shl.b64      [[P2_2_shl:%rd[0-9]+]], [[P2_2]], 16;
; CHECK-DAG:    shl.b64      [[P2_3_shl:%rd[0-9]+]], [[P2_3]], 24;
; CHECK-DAG:    or.b64       [[P2_or_0:%rd[0-9]+]], [[P2_1_shl]], [[P2_0]];
; CHECK-DAG:    or.b64       [[P2_or_1:%rd[0-9]+]], [[P2_3_shl]], [[P2_2_shl]];
; CHECK-DAG:    or.b64       [[P2_or_2:%rd[0-9]+]], [[P2_or_1]], [[P2_or_0]];
; CHECK-DAG:    shl.b64 	 [[P2_5_shl:%rd[0-9]+]], [[P2_5]], 8;
; CHECK-DAG:    shl.b64      [[P2_6_shl:%rd[0-9]+]], [[P2_6]], 16;
; CHECK-DAG:    shl.b64      [[P2_7_shl:%rd[0-9]+]], [[P2_7]], 24;
; CHECK-DAG:    or.b64       [[P2_or_3:%rd[0-9]+]], [[P2_5_shl]], [[P2_4]];
; CHECK-DAG:    or.b64       [[P2_or_4:%rd[0-9]+]], [[P2_7_shl]], [[P2_6_shl]];
; CHECK-DAG:    or.b64       [[P2_or_5:%rd[0-9]+]], [[P2_or_4]], [[P2_or_3]];
; CHECK-DAG:    shl.b64      [[P2_or_shl:%rd[0-9]+]], [[P2_or_5]], 32;
; CHECK-DAG:    or.b64       [[P2:%rd[0-9]+]], [[P2_or_shl]], [[P2_or_2]];
; CHECK-DAG:    shr.u64      [[P2_shr_1:%rd[0-9]+]], [[P2]], 8;
; CHECK-DAG:    shr.u64      [[P2_shr_2:%rd[0-9]+]], [[P2]], 16;
; CHECK-DAG:    shr.u64      [[P2_shr_3:%rd[0-9]+]], [[P2]], 24;
; CHECK-DAG:    bfe.u64      [[P2_bfe_4:%rd[0-9]+]], [[P2_or_5]], 8, 24;
; CHECK-DAG:    bfe.u64      [[P2_bfe_5:%rd[0-9]+]], [[P2_or_5]], 16, 16;
; CHECK-DAG:    bfe.u64      [[P2_bfe_6:%rd[0-9]+]], [[P2_or_5]], 24, 8;
; CHECK:        { // callseq
; CHECK:        .param .align 8 .b8 param0[32];
; CHECK-DAG:    st.param.b64 [param0],  [[P0]];
; CHECK-DAG:    st.param.b8  [param0+9],  [[P2]];
; CHECK-DAG:    st.param.b8  [param0+10], [[P2_shr_1]];
; CHECK-DAG:    st.param.b8  [param0+11], [[P2_shr_2]];
; CHECK-DAG:    st.param.b8  [param0+12], [[P2_shr_3]];
; CHECK-DAG:    st.param.b8  [param0+13], [[P2_or_5]];
; CHECK-DAG:    st.param.b8  [param0+14], [[P2_bfe_4]];
; CHECK-DAG:    st.param.b8  [param0+15], [[P2_bfe_5]];
; CHECK-DAG:    st.param.b8  [param0+16], [[P2_bfe_6]];
; CHECK:        .param .align 8 .b8 retval0[32];
; CHECK-NEXT:   call.uni (retval0),
; CHECK-NEXT:   test_s_i8f64p,
; CHECK-NEXT:   (
; CHECK-NEXT:   param0
; CHECK-NEXT:   );
; CHECK-DAG:    ld.param.b64 [[R0:%rd[0-9]+]],   [retval0];
; CHECK-DAG:    ld.param.b8  [[R2_0:%rs[0-9]+]], [retval0+9];
; CHECK-DAG:    ld.param.b8  [[R2_1:%rs[0-9]+]], [retval0+10];
; CHECK-DAG:    ld.param.b8  [[R2_2:%rs[0-9]+]], [retval0+11];
; CHECK-DAG:    ld.param.b8  [[R2_3:%rs[0-9]+]], [retval0+12];
; CHECK-DAG:    ld.param.b8  [[R2_4:%rs[0-9]+]], [retval0+13];
; CHECK-DAG:    ld.param.b8  [[R2_5:%rs[0-9]+]], [retval0+14];
; CHECK-DAG:    ld.param.b8  [[R2_6:%rs[0-9]+]], [retval0+15];
; CHECK-DAG:    ld.param.b8  [[R2_7:%rs[0-9]+]], [retval0+16];
; CHECK:        } // callseq
; CHECK-DAG:    st.param.b64 [func_retval0], [[R0]];
; CHECK-DAG:    st.param.b8  [func_retval0+9],
; CHECK-DAG:    st.param.b8  [func_retval0+10],
; CHECK-DAG:    st.param.b8  [func_retval0+11],
; CHECK-DAG:    st.param.b8  [func_retval0+12],
; CHECK-DAG:    st.param.b8  [func_retval0+13],
; CHECK-DAG:    st.param.b8  [func_retval0+14],
; CHECK-DAG:    st.param.b8  [func_retval0+15],
; CHECK-DAG:    st.param.b8  [func_retval0+16],
; CHECK:        ret;

define %s_i8f64p @test_s_i8f64p(%s_i8f64p %a) {
       %r = tail call %s_i8f64p @test_s_i8f64p(%s_i8f64p %a)
       ret %s_i8f64p %r
}
