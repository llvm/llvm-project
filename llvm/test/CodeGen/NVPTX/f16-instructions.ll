; ## Full FP16 support enabled by default.
; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_53 -asm-verbose=false \
; RUN:          -O0 -disable-post-ra -disable-fp-elim -verify-machineinstrs \
; RUN: | FileCheck -check-prefixes CHECK,CHECK-NOFTZ,CHECK-F16,CHECK-F16-NOFTZ %s
; ## Full FP16 with FTZ
; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_53 -asm-verbose=false \
; RUN:          -O0 -disable-post-ra -disable-fp-elim -verify-machineinstrs \
; RUN:          -nvptx-f32ftz \
; RUN: | FileCheck -check-prefixes CHECK,CHECK-F16,CHECK-F16-FTZ %s
; ## FP16 support explicitly disabled.
; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_53 -asm-verbose=false \
; RUN:          -O0 -disable-post-ra -disable-fp-elim --nvptx-no-f16-math \
; RUN:           -verify-machineinstrs \
; RUN: | FileCheck -check-prefixes CHECK,CHECK-NOFTZ,CHECK-NOF16 %s
; ## FP16 is not supported by hardware.
; RUN: llc < %s -O0 -mtriple=nvptx64-nvidia-cuda -mcpu=sm_52 -asm-verbose=false \
; RUN:          -disable-post-ra -disable-fp-elim -verify-machineinstrs \
; RUN: | FileCheck -check-prefixes CHECK,CHECK-NOFTZ,CHECK-NOF16 %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: test_ret_const(
; CHECK:      mov.b16         [[R:%h[0-9]+]], 0x3C00;
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define half @test_ret_const() #0 {
  ret half 1.0
}

; CHECK-LABEL: test_fadd(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fadd_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fadd_param_1];
; CHECK-F16-NOFTZ-NEXT:   add.rn.f16     [[R:%h[0-9]+]], [[A]], [[B]];
; CHECK-F16-FTZ-NEXT:   add.rn.ftz.f16     [[R:%h[0-9]+]], [[A]], [[B]];
; CHECK-NOF16-DAG:  cvt.f32.f16    [[A32:%f[0-9]+]], [[A]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[B32:%f[0-9]+]], [[B]]
; CHECK-NOF16-NEXT: add.rn.f32     [[R32:%f[0-9]+]], [[A32]], [[B32]];
; CHECK-NOF16-NEXT: cvt.rn.f16.f32 [[R:%h[0-9]+]], [[R32]]
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define half @test_fadd(half %a, half %b) #0 {
  %r = fadd half %a, %b
  ret half %r
}

; CHECK-LABEL: test_fadd_v1f16(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fadd_v1f16_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fadd_v1f16_param_1];
; CHECK-F16-NOFTZ-NEXT:   add.rn.f16     [[R:%h[0-9]+]], [[A]], [[B]];
; CHECK-F16-FTZ-NEXT:   add.rn.ftz.f16     [[R:%h[0-9]+]], [[A]], [[B]];
; CHECK-NOF16-DAG:  cvt.f32.f16    [[A32:%f[0-9]+]], [[A]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[B32:%f[0-9]+]], [[B]]
; CHECK-NOF16-NEXT: add.rn.f32     [[R32:%f[0-9]+]], [[A32]], [[B32]];
; CHECK-NOF16-NEXT: cvt.rn.f16.f32 [[R:%h[0-9]+]], [[R32]]
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <1 x half> @test_fadd_v1f16(<1 x half> %a, <1 x half> %b) #0 {
  %r = fadd <1 x half> %a, %b
  ret <1 x half> %r
}

; Check that we can lower fadd with immediate arguments.
; CHECK-LABEL: test_fadd_imm_0(
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fadd_imm_0_param_0];
; CHECK-F16-NOFTZ-DAG:    mov.b16        [[A:%h[0-9]+]], 0x3C00;
; CHECK-F16-NOFTZ-NEXT:   add.rn.f16     [[R:%h[0-9]+]], [[B]], [[A]];
; CHECK-F16-FTZ-DAG:    mov.b16        [[A:%h[0-9]+]], 0x3C00;
; CHECK-F16-FTZ-NEXT:   add.rn.ftz.f16     [[R:%h[0-9]+]], [[B]], [[A]];
; CHECK-NOF16-DAG:  cvt.f32.f16    [[B32:%f[0-9]+]], [[B]]
; CHECK-NOF16-NEXT: add.rn.f32     [[R32:%f[0-9]+]], [[B32]], 0f3F800000;
; CHECK-NOF16-NEXT: cvt.rn.f16.f32 [[R:%h[0-9]+]], [[R32]]
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define half @test_fadd_imm_0(half %b) #0 {
  %r = fadd half 1.0, %b
  ret half %r
}

; CHECK-LABEL: test_fadd_imm_1(
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fadd_imm_1_param_0];
; CHECK-F16-NOFTZ-DAG:    mov.b16        [[A:%h[0-9]+]], 0x3C00;
; CHECK-F16-NOFTZ-NEXT:   add.rn.f16     [[R:%h[0-9]+]], [[B]], [[A]];
; CHECK-F16-FTZ-DAG:    mov.b16        [[A:%h[0-9]+]], 0x3C00;
; CHECK-F16-FTZ-NEXT:   add.rn.ftz.f16     [[R:%h[0-9]+]], [[B]], [[A]];
; CHECK-NOF16-DAG:  cvt.f32.f16    [[B32:%f[0-9]+]], [[B]]
; CHECK-NOF16-NEXT: add.rn.f32     [[R32:%f[0-9]+]], [[B32]], 0f3F800000;
; CHECK-NOF16-NEXT: cvt.rn.f16.f32 [[R:%h[0-9]+]], [[R32]]
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define half @test_fadd_imm_1(half %a) #0 {
  %r = fadd half %a, 1.0
  ret half %r
}

; CHECK-LABEL: test_fsub(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fsub_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fsub_param_1];
; CHECK-F16-NOFTZ-NEXT:   sub.rn.f16     [[R:%h[0-9]+]], [[A]], [[B]];
; CHECK-F16-FTZ-NEXT:   sub.rn.ftz.f16     [[R:%h[0-9]+]], [[A]], [[B]];
; CHECK-NOF16-DAG:  cvt.f32.f16    [[A32:%f[0-9]+]], [[A]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[B32:%f[0-9]+]], [[B]]
; CHECK-NOF16-NEXT: sub.rn.f32     [[R32:%f[0-9]+]], [[A32]], [[B32]];
; CHECK-NOF16-NEXT: cvt.rn.f16.f32 [[R:%h[0-9]+]], [[R32]]
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define half @test_fsub(half %a, half %b) #0 {
  %r = fsub half %a, %b
  ret half %r
}

; CHECK-LABEL: test_fneg(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fneg_param_0];
; CHECK-F16-NOFTZ-NEXT:   mov.b16        [[Z:%h[0-9]+]], 0x0000
; CHECK-F16-NOFTZ-NEXT:   sub.rn.f16     [[R:%h[0-9]+]], [[Z]], [[A]];
; CHECK-F16-FTZ-NEXT:   mov.b16        [[Z:%h[0-9]+]], 0x0000
; CHECK-F16-FTZ-NEXT:   sub.rn.ftz.f16     [[R:%h[0-9]+]], [[Z]], [[A]];
; CHECK-NOF16-DAG:  cvt.f32.f16    [[A32:%f[0-9]+]], [[A]]
; CHECK-NOF16-DAG:  mov.f32        [[Z:%f[0-9]+]], 0f00000000;
; CHECK-NOF16-NEXT: sub.rn.f32     [[R32:%f[0-9]+]], [[Z]], [[A32]];
; CHECK-NOF16-NEXT: cvt.rn.f16.f32 [[R:%h[0-9]+]], [[R32]]
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define half @test_fneg(half %a) #0 {
  %r = fsub half 0.0, %a
  ret half %r
}

; CHECK-LABEL: test_fmul(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fmul_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fmul_param_1];
; CHECK-F16-NOFTZ-NEXT: mul.rn.f16      [[R:%h[0-9]+]], [[A]], [[B]];
; CHECK-F16-FTZ-NEXT: mul.rn.ftz.f16      [[R:%h[0-9]+]], [[A]], [[B]];
; CHECK-NOF16-DAG:  cvt.f32.f16    [[A32:%f[0-9]+]], [[A]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[B32:%f[0-9]+]], [[B]]
; CHECK-NOF16-NEXT: mul.rn.f32     [[R32:%f[0-9]+]], [[A32]], [[B32]];
; CHECK-NOF16-NEXT: cvt.rn.f16.f32 [[R:%h[0-9]+]], [[R32]]
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define half @test_fmul(half %a, half %b) #0 {
  %r = fmul half %a, %b
  ret half %r
}

; CHECK-LABEL: test_fdiv(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fdiv_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fdiv_param_1];
; CHECK-NOFTZ-DAG:  cvt.f32.f16     [[F0:%f[0-9]+]], [[A]];
; CHECK-NOFTZ-DAG:  cvt.f32.f16     [[F1:%f[0-9]+]], [[B]];
; CHECK-NOFTZ-NEXT: div.rn.f32      [[FR:%f[0-9]+]], [[F0]], [[F1]];
; CHECK-F16-FTZ-DAG:  cvt.ftz.f32.f16     [[F0:%f[0-9]+]], [[A]];
; CHECK-F16-FTZ-DAG:  cvt.ftz.f32.f16     [[F1:%f[0-9]+]], [[B]];
; CHECK-F16-FTZ-NEXT: div.rn.ftz.f32      [[FR:%f[0-9]+]], [[F0]], [[F1]];
; CHECK-NEXT: cvt.rn.f16.f32  [[R:%h[0-9]+]], [[FR]];
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define half @test_fdiv(half %a, half %b) #0 {
  %r = fdiv half %a, %b
  ret half %r
}

; CHECK-LABEL: test_frem(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_frem_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_frem_param_1];
; CHECK-NOFTZ-DAG:  cvt.f32.f16     [[FA:%f[0-9]+]], [[A]];
; CHECK-NOFTZ-DAG:  cvt.f32.f16     [[FB:%f[0-9]+]], [[B]];
; CHECK-NOFTZ-NEXT: div.rn.f32      [[D:%f[0-9]+]], [[FA]], [[FB]];
; CHECK-NOFTZ-NEXT: cvt.rmi.f32.f32 [[DI:%f[0-9]+]], [[D]];
; CHECK-NOFTZ-NEXT: mul.f32         [[RI:%f[0-9]+]], [[DI]], [[FB]];
; CHECK-NOFTZ-NEXT: sub.f32         [[RF:%f[0-9]+]], [[FA]], [[RI]];
; CHECK-F16-FTZ-DAG:  cvt.ftz.f32.f16     [[FA:%f[0-9]+]], [[A]];
; CHECK-F16-FTZ-DAG:  cvt.ftz.f32.f16     [[FB:%f[0-9]+]], [[B]];
; CHECK-F16-FTZ-NEXT: div.rn.ftz.f32      [[D:%f[0-9]+]], [[FA]], [[FB]];
; CHECK-F16-FTZ-NEXT: cvt.rmi.ftz.f32.f32 [[DI:%f[0-9]+]], [[D]];
; CHECK-F16-FTZ-NEXT: mul.ftz.f32         [[RI:%f[0-9]+]], [[DI]], [[FB]];
; CHECK-F16-FTZ-NEXT: sub.ftz.f32         [[RF:%f[0-9]+]], [[FA]], [[RI]];
; CHECK-NEXT: cvt.rn.f16.f32  [[R:%h[0-9]+]], [[RF]];
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define half @test_frem(half %a, half %b) #0 {
  %r = frem half %a, %b
  ret half %r
}

; CHECK-LABEL: test_store(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_store_param_0];
; CHECK-DAG:  ld.param.u64    %[[PTR:rd[0-9]+]], [test_store_param_1];
; CHECK-NEXT: st.b16          [%[[PTR]]], [[A]];
; CHECK-NEXT: ret;
define void @test_store(half %a, half* %b) #0 {
  store half %a, half* %b
  ret void
}

; CHECK-LABEL: test_load(
; CHECK:      ld.param.u64    %[[PTR:rd[0-9]+]], [test_load_param_0];
; CHECK-NEXT: ld.b16          [[R:%h[0-9]+]], [%[[PTR]]];
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define half @test_load(half* %a) #0 {
  %r = load half, half* %a
  ret half %r
}

; CHECK-LABEL: .visible .func test_halfp0a1(
; CHECK-DAG: ld.param.u64 %[[FROM:rd?[0-9]+]], [test_halfp0a1_param_0];
; CHECK-DAG: ld.param.u64 %[[TO:rd?[0-9]+]], [test_halfp0a1_param_1];
; CHECK-DAG: ld.u8        [[B0:%r[sd]?[0-9]+]], [%[[FROM]]]
; CHECK-DAG: st.u8        [%[[TO]]], [[B0]]
; CHECK-DAG: ld.u8        [[B1:%r[sd]?[0-9]+]], [%[[FROM]]+1]
; CHECK-DAG: st.u8        [%[[TO]]+1], [[B1]]
; CHECK: ret
define void @test_halfp0a1(half * noalias readonly %from, half * %to) {
  %1 = load half, half * %from , align 1
  store half %1, half * %to , align 1
  ret void
}

declare half @test_callee(half %a, half %b) #0

; CHECK-LABEL: test_call(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_call_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_call_param_1];
; CHECK:      {
; CHECK-DAG:  .param .b32 param0;
; CHECK-DAG:  .param .b32 param1;
; CHECK-DAG:  st.param.b16    [param0+0], [[A]];
; CHECK-DAG:  st.param.b16    [param1+0], [[B]];
; CHECK-DAG:  .param .b32 retval0;
; CHECK:      call.uni (retval0),
; CHECK-NEXT:        test_callee,
; CHECK:      );
; CHECK-NEXT: ld.param.b16    [[R:%h[0-9]+]], [retval0+0];
; CHECK-NEXT: }
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define half @test_call(half %a, half %b) #0 {
  %r = call half @test_callee(half %a, half %b)
  ret half %r
}

; CHECK-LABEL: test_call_flipped(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_call_flipped_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_call_flipped_param_1];
; CHECK:      {
; CHECK-DAG:  .param .b32 param0;
; CHECK-DAG:  .param .b32 param1;
; CHECK-DAG:  st.param.b16    [param0+0], [[B]];
; CHECK-DAG:  st.param.b16    [param1+0], [[A]];
; CHECK-DAG:  .param .b32 retval0;
; CHECK:      call.uni (retval0),
; CHECK-NEXT:        test_callee,
; CHECK:      );
; CHECK-NEXT: ld.param.b16    [[R:%h[0-9]+]], [retval0+0];
; CHECK-NEXT: }
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define half @test_call_flipped(half %a, half %b) #0 {
  %r = call half @test_callee(half %b, half %a)
  ret half %r
}

; CHECK-LABEL: test_tailcall_flipped(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_tailcall_flipped_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_tailcall_flipped_param_1];
; CHECK:      {
; CHECK-DAG:  .param .b32 param0;
; CHECK-DAG:  .param .b32 param1;
; CHECK-DAG:  st.param.b16    [param0+0], [[B]];
; CHECK-DAG:  st.param.b16    [param1+0], [[A]];
; CHECK-DAG:  .param .b32 retval0;
; CHECK:      call.uni (retval0),
; CHECK-NEXT:        test_callee,
; CHECK:      );
; CHECK-NEXT: ld.param.b16    [[R:%h[0-9]+]], [retval0+0];
; CHECK-NEXT: }
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define half @test_tailcall_flipped(half %a, half %b) #0 {
  %r = tail call half @test_callee(half %b, half %a)
  ret half %r
}

; CHECK-LABEL: test_select(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_select_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_select_param_1];
; CHECK-DAG:  setp.eq.b16     [[PRED:%p[0-9]+]], %rs{{.*}}, 1;
; CHECK-NEXT: selp.b16        [[R:%h[0-9]+]], [[A]], [[B]], [[PRED]];
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define half @test_select(half %a, half %b, i1 zeroext %c) #0 {
  %r = select i1 %c, half %a, half %b
  ret half %r
}

; CHECK-LABEL: test_select_cc(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_select_cc_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_select_cc_param_1];
; CHECK-DAG:  ld.param.b16    [[C:%h[0-9]+]], [test_select_cc_param_2];
; CHECK-DAG:  ld.param.b16    [[D:%h[0-9]+]], [test_select_cc_param_3];
; CHECK-F16-NOFTZ:  setp.neu.f16    [[PRED:%p[0-9]+]], [[C]], [[D]]
; CHECK-NOF16-DAG: cvt.f32.f16 [[DF:%f[0-9]+]], [[D]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[CF:%f[0-9]+]], [[C]];
; CHECK-NOF16: setp.neu.f32    [[PRED:%p[0-9]+]], [[CF]], [[DF]]
; CHECK:      selp.b16        [[R:%h[0-9]+]], [[A]], [[B]], [[PRED]];
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define half @test_select_cc(half %a, half %b, half %c, half %d) #0 {
  %cc = fcmp une half %c, %d
  %r = select i1 %cc, half %a, half %b
  ret half %r
}

; CHECK-LABEL: test_select_cc_f32_f16(
; CHECK-DAG:  ld.param.f32    [[A:%f[0-9]+]], [test_select_cc_f32_f16_param_0];
; CHECK-DAG:  ld.param.f32    [[B:%f[0-9]+]], [test_select_cc_f32_f16_param_1];
; CHECK-DAG:  ld.param.b16    [[C:%h[0-9]+]], [test_select_cc_f32_f16_param_2];
; CHECK-DAG:  ld.param.b16    [[D:%h[0-9]+]], [test_select_cc_f32_f16_param_3];
; CHECK-F16-NOFTZ:  setp.neu.f16    [[PRED:%p[0-9]+]], [[C]], [[D]]
; CHECK-F16-FTZ:  setp.neu.ftz.f16    [[PRED:%p[0-9]+]], [[C]], [[D]]
; CHECK-NOF16-DAG: cvt.f32.f16 [[DF:%f[0-9]+]], [[D]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[CF:%f[0-9]+]], [[C]];
; CHECK-NOF16: setp.neu.f32    [[PRED:%p[0-9]+]], [[CF]], [[DF]]
; CHECK-NEXT: selp.f32        [[R:%f[0-9]+]], [[A]], [[B]], [[PRED]];
; CHECK-NEXT: st.param.f32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define float @test_select_cc_f32_f16(float %a, float %b, half %c, half %d) #0 {
  %cc = fcmp une half %c, %d
  %r = select i1 %cc, float %a, float %b
  ret float %r
}

; CHECK-LABEL: test_select_cc_f16_f32(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_select_cc_f16_f32_param_0];
; CHECK-DAG:  ld.param.f32    [[C:%f[0-9]+]], [test_select_cc_f16_f32_param_2];
; CHECK-DAG:  ld.param.f32    [[D:%f[0-9]+]], [test_select_cc_f16_f32_param_3];
; CHECK-NOFTZ-DAG:  setp.neu.f32    [[PRED:%p[0-9]+]], [[C]], [[D]]
; CHECK-F16-FTZ-DAG:  setp.neu.ftz.f32    [[PRED:%p[0-9]+]], [[C]], [[D]]
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_select_cc_f16_f32_param_1];
; CHECK-NEXT: selp.b16        [[R:%h[0-9]+]], [[A]], [[B]], [[PRED]];
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define half @test_select_cc_f16_f32(half %a, half %b, float %c, float %d) #0 {
  %cc = fcmp une float %c, %d
  %r = select i1 %cc, half %a, half %b
  ret half %r
}

; CHECK-LABEL: test_fcmp_une(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fcmp_une_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fcmp_une_param_1];
; CHECK-F16-NOFTZ:  setp.neu.f16    [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-F16-FTZ:  setp.neu.ftz.f16    [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG: cvt.f32.f16 [[AF:%f[0-9]+]], [[A]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[BF:%f[0-9]+]], [[B]];
; CHECK-NOF16: setp.neu.f32   [[PRED:%p[0-9]+]], [[AF]], [[BF]]
; CHECK-NEXT: selp.u32        [[R:%r[0-9]+]], 1, 0, [[PRED]];
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define i1 @test_fcmp_une(half %a, half %b) #0 {
  %r = fcmp une half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_ueq(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fcmp_ueq_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fcmp_ueq_param_1];
; CHECK-F16-NOFTZ:  setp.equ.f16    [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-F16-FTZ:  setp.equ.ftz.f16    [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG: cvt.f32.f16 [[AF:%f[0-9]+]], [[A]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[BF:%f[0-9]+]], [[B]];
; CHECK-NOF16: setp.equ.f32   [[PRED:%p[0-9]+]], [[AF]], [[BF]]
; CHECK-NEXT: selp.u32        [[R:%r[0-9]+]], 1, 0, [[PRED]];
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define i1 @test_fcmp_ueq(half %a, half %b) #0 {
  %r = fcmp ueq half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_ugt(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fcmp_ugt_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fcmp_ugt_param_1];
; CHECK-F16-NOFTZ:  setp.gtu.f16    [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-F16-FTZ:  setp.gtu.ftz.f16    [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG: cvt.f32.f16 [[AF:%f[0-9]+]], [[A]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[BF:%f[0-9]+]], [[B]];
; CHECK-NOF16: setp.gtu.f32   [[PRED:%p[0-9]+]], [[AF]], [[BF]]
; CHECK-NEXT: selp.u32        [[R:%r[0-9]+]], 1, 0, [[PRED]];
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define i1 @test_fcmp_ugt(half %a, half %b) #0 {
  %r = fcmp ugt half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_uge(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fcmp_uge_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fcmp_uge_param_1];
; CHECK-F16-NOFTZ:  setp.geu.f16    [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-F16-FTZ:  setp.geu.ftz.f16    [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG: cvt.f32.f16 [[AF:%f[0-9]+]], [[A]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[BF:%f[0-9]+]], [[B]];
; CHECK-NOF16: setp.geu.f32   [[PRED:%p[0-9]+]], [[AF]], [[BF]]
; CHECK-NEXT: selp.u32        [[R:%r[0-9]+]], 1, 0, [[PRED]];
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define i1 @test_fcmp_uge(half %a, half %b) #0 {
  %r = fcmp uge half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_ult(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fcmp_ult_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fcmp_ult_param_1];
; CHECK-F16-NOFTZ:  setp.ltu.f16    [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-F16-FTZ:  setp.ltu.ftz.f16    [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG: cvt.f32.f16 [[AF:%f[0-9]+]], [[A]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[BF:%f[0-9]+]], [[B]];
; CHECK-NOF16: setp.ltu.f32   [[PRED:%p[0-9]+]], [[AF]], [[BF]]
; CHECK-NEXT: selp.u32        [[R:%r[0-9]+]], 1, 0, [[PRED]];
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define i1 @test_fcmp_ult(half %a, half %b) #0 {
  %r = fcmp ult half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_ule(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fcmp_ule_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fcmp_ule_param_1];
; CHECK-F16-NOFTZ:  setp.leu.f16    [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-F16-FTZ:  setp.leu.ftz.f16    [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG: cvt.f32.f16 [[AF:%f[0-9]+]], [[A]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[BF:%f[0-9]+]], [[B]];
; CHECK-NOF16: setp.leu.f32   [[PRED:%p[0-9]+]], [[AF]], [[BF]]
; CHECK-NEXT: selp.u32        [[R:%r[0-9]+]], 1, 0, [[PRED]];
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define i1 @test_fcmp_ule(half %a, half %b) #0 {
  %r = fcmp ule half %a, %b
  ret i1 %r
}


; CHECK-LABEL: test_fcmp_uno(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fcmp_uno_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fcmp_uno_param_1];
; CHECK-F16-NOFTZ:  setp.nan.f16    [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-F16-FTZ:  setp.nan.ftz.f16    [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG: cvt.f32.f16 [[AF:%f[0-9]+]], [[A]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[BF:%f[0-9]+]], [[B]];
; CHECK-NOF16: setp.nan.f32   [[PRED:%p[0-9]+]], [[AF]], [[BF]]
; CHECK-NEXT: selp.u32        [[R:%r[0-9]+]], 1, 0, [[PRED]];
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define i1 @test_fcmp_uno(half %a, half %b) #0 {
  %r = fcmp uno half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_one(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fcmp_one_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fcmp_one_param_1];
; CHECK-F16-NOFTZ:  setp.ne.f16     [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-F16-FTZ:  setp.ne.ftz.f16     [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG: cvt.f32.f16 [[AF:%f[0-9]+]], [[A]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[BF:%f[0-9]+]], [[B]];
; CHECK-NOF16: setp.ne.f32    [[PRED:%p[0-9]+]], [[AF]], [[BF]]
; CHECK-NEXT: selp.u32        [[R:%r[0-9]+]], 1, 0, [[PRED]];
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define i1 @test_fcmp_one(half %a, half %b) #0 {
  %r = fcmp one half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_oeq(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fcmp_oeq_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fcmp_oeq_param_1];
; CHECK-F16-NOFTZ:  setp.eq.f16     [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-F16-FTZ:  setp.eq.ftz.f16     [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG: cvt.f32.f16 [[AF:%f[0-9]+]], [[A]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[BF:%f[0-9]+]], [[B]];
; CHECK-NOF16: setp.eq.f32    [[PRED:%p[0-9]+]], [[AF]], [[BF]]
; CHECK-NEXT: selp.u32        [[R:%r[0-9]+]], 1, 0, [[PRED]];
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define i1 @test_fcmp_oeq(half %a, half %b) #0 {
  %r = fcmp oeq half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_ogt(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fcmp_ogt_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fcmp_ogt_param_1];
; CHECK-F16-NOFTZ:  setp.gt.f16     [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-F16-FTZ:  setp.gt.ftz.f16     [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG: cvt.f32.f16 [[AF:%f[0-9]+]], [[A]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[BF:%f[0-9]+]], [[B]];
; CHECK-NOF16: setp.gt.f32    [[PRED:%p[0-9]+]], [[AF]], [[BF]]
; CHECK-NEXT: selp.u32        [[R:%r[0-9]+]], 1, 0, [[PRED]];
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define i1 @test_fcmp_ogt(half %a, half %b) #0 {
  %r = fcmp ogt half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_oge(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fcmp_oge_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fcmp_oge_param_1];
; CHECK-F16-NOFTZ:  setp.ge.f16     [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-F16-FTZ:  setp.ge.ftz.f16     [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG: cvt.f32.f16 [[AF:%f[0-9]+]], [[A]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[BF:%f[0-9]+]], [[B]];
; CHECK-NOF16: setp.ge.f32    [[PRED:%p[0-9]+]], [[AF]], [[BF]]
; CHECK-NEXT: selp.u32        [[R:%r[0-9]+]], 1, 0, [[PRED]];
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define i1 @test_fcmp_oge(half %a, half %b) #0 {
  %r = fcmp oge half %a, %b
  ret i1 %r
}

; XCHECK-LABEL: test_fcmp_olt(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fcmp_olt_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fcmp_olt_param_1];
; CHECK-F16-NOFTZ:  setp.lt.f16     [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-F16-FTZ:  setp.lt.ftz.f16     [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG: cvt.f32.f16 [[AF:%f[0-9]+]], [[A]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[BF:%f[0-9]+]], [[B]];
; CHECK-NOF16: setp.lt.f32    [[PRED:%p[0-9]+]], [[AF]], [[BF]]
; CHECK-NEXT: selp.u32        [[R:%r[0-9]+]], 1, 0, [[PRED]];
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define i1 @test_fcmp_olt(half %a, half %b) #0 {
  %r = fcmp olt half %a, %b
  ret i1 %r
}

; XCHECK-LABEL: test_fcmp_ole(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fcmp_ole_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fcmp_ole_param_1];
; CHECK-F16-NOFTZ:  setp.le.f16     [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-F16-FTZ:  setp.le.ftz.f16     [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG: cvt.f32.f16 [[AF:%f[0-9]+]], [[A]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[BF:%f[0-9]+]], [[B]];
; CHECK-NOF16: setp.le.f32    [[PRED:%p[0-9]+]], [[AF]], [[BF]]
; CHECK-NEXT: selp.u32        [[R:%r[0-9]+]], 1, 0, [[PRED]];
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define i1 @test_fcmp_ole(half %a, half %b) #0 {
  %r = fcmp ole half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_ord(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fcmp_ord_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fcmp_ord_param_1];
; CHECK-F16-NOFTZ:  setp.num.f16    [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-F16-FTZ:  setp.num.ftz.f16    [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG: cvt.f32.f16 [[AF:%f[0-9]+]], [[A]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[BF:%f[0-9]+]], [[B]];
; CHECK-NOF16: setp.num.f32   [[PRED:%p[0-9]+]], [[AF]], [[BF]]
; CHECK-NEXT: selp.u32        [[R:%r[0-9]+]], 1, 0, [[PRED]];
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define i1 @test_fcmp_ord(half %a, half %b) #0 {
  %r = fcmp ord half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_br_cc(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_br_cc_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_br_cc_param_1];
; CHECK-DAG:  ld.param.u64    %[[C:rd[0-9]+]], [test_br_cc_param_2];
; CHECK-DAG:  ld.param.u64    %[[D:rd[0-9]+]], [test_br_cc_param_3];
; CHECK-F16-NOFTZ:  setp.lt.f16     [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-F16-FTZ:  setp.lt.ftz.f16     [[PRED:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG: cvt.f32.f16 [[AF:%f[0-9]+]], [[A]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[BF:%f[0-9]+]], [[B]];
; CHECK-NOF16: setp.lt.f32    [[PRED:%p[0-9]+]], [[AF]], [[BF]]
; CHECK-NEXT: @[[PRED]] bra   [[LABEL:LBB.*]];
; CHECK:      st.u32  [%[[C]]],
; CHECK:      [[LABEL]]:
; CHECK:      st.u32  [%[[D]]],
; CHECK:      ret;
define void @test_br_cc(half %a, half %b, i32* %p1, i32* %p2) #0 {
  %c = fcmp uge half %a, %b
  br i1 %c, label %then, label %else
then:
  store i32 0, i32* %p1
  ret void
else:
  store i32 0, i32* %p2
  ret void
}

; CHECK-LABEL: test_phi(
; CHECK:      ld.param.u64    %[[P1:rd[0-9]+]], [test_phi_param_0];
; CHECK:      ld.b16  {{%h[0-9]+}}, [%[[P1]]];
; CHECK: [[LOOP:LBB[0-9_]+]]:
; CHECK:      mov.b16 [[R:%h[0-9]+]], [[AB:%h[0-9]+]];
; CHECK:      ld.b16  [[AB:%h[0-9]+]], [%[[P1]]];
; CHECK:      {
; CHECK:      st.param.b64    [param0+0], %[[P1]];
; CHECK:      call.uni (retval0),
; CHECK-NEXT: test_dummy
; CHECK:      }
; CHECK:      setp.eq.b32     [[PRED:%p[0-9]+]], %r{{[0-9]+}}, 1;
; CHECK:      @[[PRED]] bra   [[LOOP]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_phi(half* %p1) #0 {
entry:
  %a = load half, half* %p1
  br label %loop
loop:
  %r = phi half [%a, %entry], [%b, %loop]
  %b = load half, half* %p1
  %c = call i1 @test_dummy(half* %p1)
  br i1 %c, label %loop, label %return
return:
  ret half %r
}
declare i1 @test_dummy(half* %p1) #0

; CHECK-LABEL: test_fptosi_i32(
; CHECK:      ld.param.b16    [[A:%h[0-9]+]], [test_fptosi_i32_param_0];
; CHECK:      cvt.rzi.s32.f16 [[R:%r[0-9]+]], [[A]];
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define i32 @test_fptosi_i32(half %a) #0 {
  %r = fptosi half %a to i32
  ret i32 %r
}

; CHECK-LABEL: test_fptosi_i64(
; CHECK:      ld.param.b16    [[A:%h[0-9]+]], [test_fptosi_i64_param_0];
; CHECK:      cvt.rzi.s64.f16 [[R:%rd[0-9]+]], [[A]];
; CHECK:      st.param.b64    [func_retval0+0], [[R]];
; CHECK:      ret;
define i64 @test_fptosi_i64(half %a) #0 {
  %r = fptosi half %a to i64
  ret i64 %r
}

; CHECK-LABEL: test_fptoui_i32(
; CHECK:      ld.param.b16    [[A:%h[0-9]+]], [test_fptoui_i32_param_0];
; CHECK:      cvt.rzi.u32.f16 [[R:%r[0-9]+]], [[A]];
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define i32 @test_fptoui_i32(half %a) #0 {
  %r = fptoui half %a to i32
  ret i32 %r
}

; CHECK-LABEL: test_fptoui_i64(
; CHECK:      ld.param.b16    [[A:%h[0-9]+]], [test_fptoui_i64_param_0];
; CHECK:      cvt.rzi.u64.f16 [[R:%rd[0-9]+]], [[A]];
; CHECK:      st.param.b64    [func_retval0+0], [[R]];
; CHECK:      ret;
define i64 @test_fptoui_i64(half %a) #0 {
  %r = fptoui half %a to i64
  ret i64 %r
}

; CHECK-LABEL: test_uitofp_i32(
; CHECK:      ld.param.u32    [[A:%r[0-9]+]], [test_uitofp_i32_param_0];
; CHECK:      cvt.rn.f16.u32  [[R:%h[0-9]+]], [[A]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_uitofp_i32(i32 %a) #0 {
  %r = uitofp i32 %a to half
  ret half %r
}

; CHECK-LABEL: test_uitofp_i64(
; CHECK:      ld.param.u64    [[A:%rd[0-9]+]], [test_uitofp_i64_param_0];
; CHECK:      cvt.rn.f16.u64  [[R:%h[0-9]+]], [[A]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_uitofp_i64(i64 %a) #0 {
  %r = uitofp i64 %a to half
  ret half %r
}

; CHECK-LABEL: test_sitofp_i32(
; CHECK:      ld.param.u32    [[A:%r[0-9]+]], [test_sitofp_i32_param_0];
; CHECK:      cvt.rn.f16.s32  [[R:%h[0-9]+]], [[A]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_sitofp_i32(i32 %a) #0 {
  %r = sitofp i32 %a to half
  ret half %r
}

; CHECK-LABEL: test_sitofp_i64(
; CHECK:      ld.param.u64    [[A:%rd[0-9]+]], [test_sitofp_i64_param_0];
; CHECK:      cvt.rn.f16.s64  [[R:%h[0-9]+]], [[A]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_sitofp_i64(i64 %a) #0 {
  %r = sitofp i64 %a to half
  ret half %r
}

; CHECK-LABEL: test_uitofp_i32_fadd(
; CHECK-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_uitofp_i32_fadd_param_0];
; CHECK-DAG:  cvt.rn.f16.u32  [[C:%h[0-9]+]], [[A]];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_uitofp_i32_fadd_param_1];
; CHECK-F16-NOFTZ:       add.rn.f16      [[R:%h[0-9]+]], [[B]], [[C]];
; CHECK-F16-FTZ:       add.rn.ftz.f16      [[R:%h[0-9]+]], [[B]], [[C]];
; CHECK-NOF16-DAG:  cvt.f32.f16    [[B32:%f[0-9]+]], [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[C32:%f[0-9]+]], [[C]]
; CHECK-NOF16-NEXT: add.rn.f32     [[R32:%f[0-9]+]], [[B32]], [[C32]];
; CHECK-NOF16-NEXT: cvt.rn.f16.f32 [[R:%h[0-9]+]], [[R32]]
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_uitofp_i32_fadd(i32 %a, half %b) #0 {
  %c = uitofp i32 %a to half
  %r = fadd half %b, %c
  ret half %r
}

; CHECK-LABEL: test_sitofp_i32_fadd(
; CHECK-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_sitofp_i32_fadd_param_0];
; CHECK-DAG:  cvt.rn.f16.s32  [[C:%h[0-9]+]], [[A]];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_sitofp_i32_fadd_param_1];
; CHECK-F16-NOFTZ:         add.rn.f16     [[R:%h[0-9]+]], [[B]], [[C]];
; CHECK-F16-FTZ:         add.rn.ftz.f16     [[R:%h[0-9]+]], [[B]], [[C]];
; XCHECK-NOF16-DAG:  cvt.f32.f16    [[B32:%f[0-9]+]], [[B]]
; XCHECK-NOF16-DAG:  cvt.f32.f16    [[C32:%f[0-9]+]], [[C]]
; XCHECK-NOF16-NEXT: add.rn.f32     [[R32:%f[0-9]+]], [[B32]], [[C32]];
; XCHECK-NOF16-NEXT: cvt.rn.f16.f32 [[R:%h[0-9]+]], [[R32]]
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_sitofp_i32_fadd(i32 %a, half %b) #0 {
  %c = sitofp i32 %a to half
  %r = fadd half %b, %c
  ret half %r
}

; CHECK-LABEL: test_fptrunc_float(
; CHECK:      ld.param.f32    [[A:%f[0-9]+]], [test_fptrunc_float_param_0];
; CHECK:      cvt.rn.f16.f32  [[R:%h[0-9]+]], [[A]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_fptrunc_float(float %a) #0 {
  %r = fptrunc float %a to half
  ret half %r
}

; CHECK-LABEL: test_fptrunc_double(
; CHECK:      ld.param.f64    [[A:%fd[0-9]+]], [test_fptrunc_double_param_0];
; CHECK:      cvt.rn.f16.f64  [[R:%h[0-9]+]], [[A]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_fptrunc_double(double %a) #0 {
  %r = fptrunc double %a to half
  ret half %r
}

; CHECK-LABEL: test_fpext_float(
; CHECK:      ld.param.b16    [[A:%h[0-9]+]], [test_fpext_float_param_0];
; CHECK-NOFTZ:      cvt.f32.f16     [[R:%f[0-9]+]], [[A]];
; CHECK-F16-FTZ:      cvt.ftz.f32.f16     [[R:%f[0-9]+]], [[A]];
; CHECK:      st.param.f32    [func_retval0+0], [[R]];
; CHECK:      ret;
define float @test_fpext_float(half %a) #0 {
  %r = fpext half %a to float
  ret float %r
}

; CHECK-LABEL: test_fpext_double(
; CHECK:      ld.param.b16    [[A:%h[0-9]+]], [test_fpext_double_param_0];
; CHECK:      cvt.f64.f16     [[R:%fd[0-9]+]], [[A]];
; CHECK:      st.param.f64    [func_retval0+0], [[R]];
; CHECK:      ret;
define double @test_fpext_double(half %a) #0 {
  %r = fpext half %a to double
  ret double %r
}


; CHECK-LABEL: test_bitcast_halftoi16(
; CHECK:      ld.param.b16    [[AH:%h[0-9]+]], [test_bitcast_halftoi16_param_0];
; CHECK:      mov.b16         [[AS:%rs[0-9]+]], [[AH]]
; CHECK:      cvt.u32.u16     [[R:%r[0-9]+]], [[AS]]
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define i16 @test_bitcast_halftoi16(half %a) #0 {
  %r = bitcast half %a to i16
  ret i16 %r
}

; CHECK-LABEL: test_bitcast_i16tohalf(
; CHECK:      ld.param.u16    [[AS:%rs[0-9]+]], [test_bitcast_i16tohalf_param_0];
; CHECK:      mov.b16         [[AH:%h[0-9]+]], [[AS]]
; CHECK:      st.param.b16    [func_retval0+0], [[AH]];
; CHECK:      ret;
define half @test_bitcast_i16tohalf(i16 %a) #0 {
  %r = bitcast i16 %a to half
  ret half %r
}


declare half @llvm.sqrt.f16(half %a) #0
declare half @llvm.powi.f16(half %a, i32 %b) #0
declare half @llvm.sin.f16(half %a) #0
declare half @llvm.cos.f16(half %a) #0
declare half @llvm.pow.f16(half %a, half %b) #0
declare half @llvm.exp.f16(half %a) #0
declare half @llvm.exp2.f16(half %a) #0
declare half @llvm.log.f16(half %a) #0
declare half @llvm.log10.f16(half %a) #0
declare half @llvm.log2.f16(half %a) #0
declare half @llvm.fma.f16(half %a, half %b, half %c) #0
declare half @llvm.fabs.f16(half %a) #0
declare half @llvm.minnum.f16(half %a, half %b) #0
declare half @llvm.maxnum.f16(half %a, half %b) #0
declare half @llvm.copysign.f16(half %a, half %b) #0
declare half @llvm.floor.f16(half %a) #0
declare half @llvm.ceil.f16(half %a) #0
declare half @llvm.trunc.f16(half %a) #0
declare half @llvm.rint.f16(half %a) #0
declare half @llvm.nearbyint.f16(half %a) #0
declare half @llvm.round.f16(half %a) #0
declare half @llvm.fmuladd.f16(half %a, half %b, half %c) #0

; CHECK-LABEL: test_sqrt(
; CHECK:      ld.param.b16    [[A:%h[0-9]+]], [test_sqrt_param_0];
; CHECK-NOFTZ:      cvt.f32.f16     [[AF:%f[0-9]+]], [[A]];
; CHECK-NOFTZ:      sqrt.rn.f32     [[RF:%f[0-9]+]], [[AF]];
; CHECK-F16-FTZ:      cvt.ftz.f32.f16     [[AF:%f[0-9]+]], [[A]];
; CHECK-F16-FTZ:      sqrt.rn.ftz.f32     [[RF:%f[0-9]+]], [[AF]];
; CHECK:      cvt.rn.f16.f32  [[R:%h[0-9]+]], [[RF]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_sqrt(half %a) #0 {
  %r = call half @llvm.sqrt.f16(half %a)
  ret half %r
}

;;; Can't do this yet: requires libcall.
; XCHECK-LABEL: test_powi(
;define half @test_powi(half %a, i32 %b) #0 {
;  %r = call half @llvm.powi.f16(half %a, i32 %b)
;  ret half %r
;}

; CHECK-LABEL: test_sin(
; CHECK:      ld.param.b16    [[A:%h[0-9]+]], [test_sin_param_0];
; CHECK-NOFTZ:      cvt.f32.f16     [[AF:%f[0-9]+]], [[A]];
; CHECK-F16-FTZ:      cvt.ftz.f32.f16     [[AF:%f[0-9]+]], [[A]];
; CHECK:      sin.approx.f32  [[RF:%f[0-9]+]], [[AF]];
; CHECK:      cvt.rn.f16.f32  [[R:%h[0-9]+]], [[RF]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_sin(half %a) #0 #1 {
  %r = call half @llvm.sin.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_cos(
; CHECK:      ld.param.b16    [[A:%h[0-9]+]], [test_cos_param_0];
; CHECK-NOFTZ:      cvt.f32.f16     [[AF:%f[0-9]+]], [[A]];
; CHECK-F16-FTZ:      cvt.ftz.f32.f16     [[AF:%f[0-9]+]], [[A]];
; CHECK:      cos.approx.f32  [[RF:%f[0-9]+]], [[AF]];
; CHECK:      cvt.rn.f16.f32  [[R:%h[0-9]+]], [[RF]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_cos(half %a) #0 #1 {
  %r = call half @llvm.cos.f16(half %a)
  ret half %r
}

;;; Can't do this yet: requires libcall.
; XCHECK-LABEL: test_pow(
;define half @test_pow(half %a, half %b) #0 {
;  %r = call half @llvm.pow.f16(half %a, half %b)
;  ret half %r
;}

;;; Can't do this yet: requires libcall.
; XCHECK-LABEL: test_exp(
;define half @test_exp(half %a) #0 {
;  %r = call half @llvm.exp.f16(half %a)
;  ret half %r
;}

;;; Can't do this yet: requires libcall.
; XCHECK-LABEL: test_exp2(
;define half @test_exp2(half %a) #0 {
;  %r = call half @llvm.exp2.f16(half %a)
;  ret half %r
;}

;;; Can't do this yet: requires libcall.
; XCHECK-LABEL: test_log(
;define half @test_log(half %a) #0 {
;  %r = call half @llvm.log.f16(half %a)
;  ret half %r
;}

;;; Can't do this yet: requires libcall.
; XCHECK-LABEL: test_log10(
;define half @test_log10(half %a) #0 {
;  %r = call half @llvm.log10.f16(half %a)
;  ret half %r
;}

;;; Can't do this yet: requires libcall.
; XCHECK-LABEL: test_log2(
;define half @test_log2(half %a) #0 {
;  %r = call half @llvm.log2.f16(half %a)
;  ret half %r
;}

; CHECK-LABEL: test_fma(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fma_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fma_param_1];
; CHECK-DAG:  ld.param.b16    [[C:%h[0-9]+]], [test_fma_param_2];
; CHECK-F16-NOFTZ:      fma.rn.f16      [[R:%h[0-9]+]], [[A]], [[B]], [[C]];
; CHECK-F16-FTZ:      fma.rn.ftz.f16      [[R:%h[0-9]+]], [[A]], [[B]], [[C]];
; CHECK-NOF16-DAG:  cvt.f32.f16    [[A32:%f[0-9]+]], [[A]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[B32:%f[0-9]+]], [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[C32:%f[0-9]+]], [[C]]
; CHECK-NOF16-NEXT: fma.rn.f32     [[R32:%f[0-9]+]], [[A32]], [[B32]], [[C32]];
; CHECK-NOF16-NEXT: cvt.rn.f16.f32 [[R:%h[0-9]+]], [[R32]]
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret
define half @test_fma(half %a, half %b, half %c) #0 {
  %r = call half @llvm.fma.f16(half %a, half %b, half %c)
  ret half %r
}

; CHECK-LABEL: test_fabs(
; CHECK:      ld.param.b16    [[A:%h[0-9]+]], [test_fabs_param_0];
; CHECK-NOFTZ:      cvt.f32.f16     [[AF:%f[0-9]+]], [[A]];
; CHECK-NOFTZ:      abs.f32         [[RF:%f[0-9]+]], [[AF]];
; CHECK-F16-FTZ:      cvt.ftz.f32.f16     [[AF:%f[0-9]+]], [[A]];
; CHECK-F16-FTZ:      abs.ftz.f32         [[RF:%f[0-9]+]], [[AF]];
; CHECK:      cvt.rn.f16.f32  [[R:%h[0-9]+]], [[RF]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_fabs(half %a) #0 {
  %r = call half @llvm.fabs.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_minnum(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_minnum_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_minnum_param_1];
; CHECK-NOFTZ-DAG:  cvt.f32.f16     [[AF:%f[0-9]+]], [[A]];
; CHECK-NOFTZ-DAG:  cvt.f32.f16     [[BF:%f[0-9]+]], [[B]];
; CHECK-NOFTZ:      min.f32         [[RF:%f[0-9]+]], [[AF]], [[BF]];
; CHECK-F16-FTZ-DAG:  cvt.ftz.f32.f16     [[AF:%f[0-9]+]], [[A]];
; CHECK-F16-FTZ-DAG:  cvt.ftz.f32.f16     [[BF:%f[0-9]+]], [[B]];
; CHECK-F16-FTZ:      min.ftz.f32         [[RF:%f[0-9]+]], [[AF]], [[BF]];
; CHECK:      cvt.rn.f16.f32  [[R:%h[0-9]+]], [[RF]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_minnum(half %a, half %b) #0 {
  %r = call half @llvm.minnum.f16(half %a, half %b)
  ret half %r
}

; CHECK-LABEL: test_maxnum(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_maxnum_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_maxnum_param_1];
; CHECK-NOFTZ-DAG:  cvt.f32.f16     [[AF:%f[0-9]+]], [[A]];
; CHECK-NOFTZ-DAG:  cvt.f32.f16     [[BF:%f[0-9]+]], [[B]];
; CHECK-NOFTZ:      max.f32         [[RF:%f[0-9]+]], [[AF]], [[BF]];
; CHECK-F16-FTZ-DAG:  cvt.ftz.f32.f16     [[AF:%f[0-9]+]], [[A]];
; CHECK-F16-FTZ-DAG:  cvt.ftz.f32.f16     [[BF:%f[0-9]+]], [[B]];
; CHECK-F16-FTZ:      max.ftz.f32         [[RF:%f[0-9]+]], [[AF]], [[BF]];
; CHECK:      cvt.rn.f16.f32  [[R:%h[0-9]+]], [[RF]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_maxnum(half %a, half %b) #0 {
  %r = call half @llvm.maxnum.f16(half %a, half %b)
  ret half %r
}

; CHECK-LABEL: test_copysign(
; CHECK-DAG:  ld.param.b16    [[AH:%h[0-9]+]], [test_copysign_param_0];
; CHECK-DAG:  ld.param.b16    [[BH:%h[0-9]+]], [test_copysign_param_1];
; CHECK-DAG:  mov.b16         [[AS:%rs[0-9]+]], [[AH]];
; CHECK-DAG:  mov.b16         [[BS:%rs[0-9]+]], [[BH]];
; CHECK-DAG:  and.b16         [[AX:%rs[0-9]+]], [[AS]], 32767;
; CHECK-DAG:  and.b16         [[BX:%rs[0-9]+]], [[BS]], -32768;
; CHECK:      or.b16          [[RX:%rs[0-9]+]], [[AX]], [[BX]];
; CHECK:      mov.b16         [[R:%h[0-9]+]], [[RX]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_copysign(half %a, half %b) #0 {
  %r = call half @llvm.copysign.f16(half %a, half %b)
  ret half %r
}

; CHECK-LABEL: test_copysign_f32(
; CHECK-DAG:  ld.param.b16    [[AH:%h[0-9]+]], [test_copysign_f32_param_0];
; CHECK-DAG:  ld.param.f32    [[BF:%f[0-9]+]], [test_copysign_f32_param_1];
; CHECK-DAG:  mov.b16         [[A:%rs[0-9]+]], [[AH]];
; CHECK-DAG:  mov.b32         [[B:%r[0-9]+]], [[BF]];
; CHECK-DAG:  and.b16         [[AX:%rs[0-9]+]], [[A]], 32767;
; CHECK-DAG:  and.b32         [[BX0:%r[0-9]+]], [[B]], -2147483648;
; CHECK-DAG:  shr.u32         [[BX1:%r[0-9]+]], [[BX0]], 16;
; CHECK-DAG:  cvt.u16.u32     [[BX2:%rs[0-9]+]], [[BX1]];
; CHECK:      or.b16          [[RX:%rs[0-9]+]], [[AX]], [[BX2]];
; CHECK:      mov.b16         [[R:%h[0-9]+]], [[RX]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_copysign_f32(half %a, float %b) #0 {
  %tb = fptrunc float %b to half
  %r = call half @llvm.copysign.f16(half %a, half %tb)
  ret half %r
}

; CHECK-LABEL: test_copysign_f64(
; CHECK-DAG:  ld.param.b16    [[AH:%h[0-9]+]], [test_copysign_f64_param_0];
; CHECK-DAG:  ld.param.f64    [[BD:%fd[0-9]+]], [test_copysign_f64_param_1];
; CHECK-DAG:  mov.b16         [[A:%rs[0-9]+]], [[AH]];
; CHECK-DAG:  mov.b64         [[B:%rd[0-9]+]], [[BD]];
; CHECK-DAG:  and.b16         [[AX:%rs[0-9]+]], [[A]], 32767;
; CHECK-DAG:  and.b64         [[BX0:%rd[0-9]+]], [[B]], -9223372036854775808;
; CHECK-DAG:  shr.u64         [[BX1:%rd[0-9]+]], [[BX0]], 48;
; CHECK-DAG:  cvt.u16.u64     [[BX2:%rs[0-9]+]], [[BX1]];
; CHECK:      or.b16          [[RX:%rs[0-9]+]], [[AX]], [[BX2]];
; CHECK:      mov.b16         [[R:%h[0-9]+]], [[RX]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_copysign_f64(half %a, double %b) #0 {
  %tb = fptrunc double %b to half
  %r = call half @llvm.copysign.f16(half %a, half %tb)
  ret half %r
}

; CHECK-LABEL: test_copysign_extended(
; CHECK-DAG:  ld.param.b16    [[AH:%h[0-9]+]], [test_copysign_extended_param_0];
; CHECK-DAG:  ld.param.b16    [[BH:%h[0-9]+]], [test_copysign_extended_param_1];
; CHECK-DAG:  mov.b16         [[AS:%rs[0-9]+]], [[AH]];
; CHECK-DAG:  mov.b16         [[BS:%rs[0-9]+]], [[BH]];
; CHECK-DAG:  and.b16         [[AX:%rs[0-9]+]], [[AS]], 32767;
; CHECK-DAG:  and.b16         [[BX:%rs[0-9]+]], [[BS]], -32768;
; CHECK:      or.b16          [[RX:%rs[0-9]+]], [[AX]], [[BX]];
; CHECK:      mov.b16         [[R:%h[0-9]+]], [[RX]];
; CHECK-NOFTZ: cvt.f32.f16     [[XR:%f[0-9]+]], [[R]];
; CHECK-F16-FTZ:   cvt.ftz.f32.f16 [[XR:%f[0-9]+]], [[R]];
; CHECK:      st.param.f32    [func_retval0+0], [[XR]];
; CHECK:      ret;
define float @test_copysign_extended(half %a, half %b) #0 {
  %r = call half @llvm.copysign.f16(half %a, half %b)
  %xr = fpext half %r to float
  ret float %xr
}

; CHECK-LABEL: test_floor(
; CHECK:      ld.param.b16    [[A:%h[0-9]+]], [test_floor_param_0];
; CHECK:      cvt.rmi.f16.f16 [[R:%h[0-9]+]], [[A]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_floor(half %a) #0 {
  %r = call half @llvm.floor.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_ceil(
; CHECK:      ld.param.b16    [[A:%h[0-9]+]], [test_ceil_param_0];
; CHECK:      cvt.rpi.f16.f16 [[R:%h[0-9]+]], [[A]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_ceil(half %a) #0 {
  %r = call half @llvm.ceil.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_trunc(
; CHECK:      ld.param.b16    [[A:%h[0-9]+]], [test_trunc_param_0];
; CHECK:      cvt.rzi.f16.f16 [[R:%h[0-9]+]], [[A]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_trunc(half %a) #0 {
  %r = call half @llvm.trunc.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_rint(
; CHECK:      ld.param.b16    [[A:%h[0-9]+]], [test_rint_param_0];
; CHECK:      cvt.rni.f16.f16 [[R:%h[0-9]+]], [[A]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_rint(half %a) #0 {
  %r = call half @llvm.rint.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_nearbyint(
; CHECK:      ld.param.b16    [[A:%h[0-9]+]], [test_nearbyint_param_0];
; CHECK:      cvt.rni.f16.f16 [[R:%h[0-9]+]], [[A]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_nearbyint(half %a) #0 {
  %r = call half @llvm.nearbyint.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_round(
; CHECK:      ld.param.b16    [[A:%h[0-9]+]], [test_round_param_0];
; CHECK:      cvt.rni.f16.f16 [[R:%h[0-9]+]], [[A]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_round(half %a) #0 {
  %r = call half @llvm.round.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_fmuladd(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fmuladd_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fmuladd_param_1];
; CHECK-DAG:  ld.param.b16    [[C:%h[0-9]+]], [test_fmuladd_param_2];
; CHECK-F16-NOFTZ:        fma.rn.f16     [[R:%h[0-9]+]], [[A]], [[B]], [[C]];
; CHECK-F16-FTZ:        fma.rn.ftz.f16     [[R:%h[0-9]+]], [[A]], [[B]], [[C]];
; CHECK-NOF16-DAG:  cvt.f32.f16    [[A32:%f[0-9]+]], [[A]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[B32:%f[0-9]+]], [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[C32:%f[0-9]+]], [[C]]
; CHECK-NOF16-NEXT: fma.rn.f32     [[R32:%f[0-9]+]], [[A32]], [[B32]], [[C32]];
; CHECK-NOF16-NEXT: cvt.rn.f16.f32 [[R:%h[0-9]+]], [[R32]]
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_fmuladd(half %a, half %b, half %c) #0 {
  %r = call half @llvm.fmuladd.f16(half %a, half %b, half %c)
  ret half %r
}

attributes #0 = { nounwind }
attributes #1 = { "unsafe-fp-math" = "true" }
