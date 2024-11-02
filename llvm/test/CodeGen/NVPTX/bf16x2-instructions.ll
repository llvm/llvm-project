; RUN: llc < %s -march=nvptx64 -mcpu=sm_80 -mattr=+ptx71 | FileCheck --check-prefixes=CHECK,SM80 %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_90 -mattr=+ptx78 | FileCheck --check-prefixes=CHECK,SM90 %s
; RUN: %if ptxas-11.8 %{ llc < %s -march=nvptx64 -mcpu=sm_80 -mattr=+ptx71 | %ptxas-verify -arch=sm_80 %}
; RUN: %if ptxas-11.8 %{ llc < %s -march=nvptx64 -mcpu=sm_90 -mattr=+ptx78 | %ptxas-verify -arch=sm_90 %}

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: test_ret_const(
; CHECK:     mov.b32         [[T:%r[0-9+]]], 1073758080;
; CHECK:     st.param.b32    [func_retval0+0], [[T]];
; CHECK-NEXT: ret;

define <2 x bfloat> @test_ret_const() #0 {
  ret <2 x bfloat> <bfloat 1.0, bfloat 2.0>
}

; Check that we can lower fadd with immediate arguments.
; CHECK-LABEL: test_fadd_imm_0(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fadd_imm_0_param_0];
;
; SM90-DAG:        mov.b32        [[I:%r[0-9+]]], 1073758080;
; SM90-DAG:        add.rn.bf16x2   [[R:%r[0-9]+]], [[A]], [[I]];
;
; SM80-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; SM80-DAG:  cvt.f32.bf16    [[FA0:%f[0-9]+]], [[A0]]
; SM80-DAG:  cvt.f32.bf16    [[FA1:%f[0-9]+]], [[A1]]
; SM80-DAG:  add.rn.f32     [[FR0:%f[0-9]+]], [[FA0]], 0f3F800000;
; SM80-DAG:  add.rn.f32     [[FR1:%f[0-9]+]], [[FA1]], 0f40000000;
; SM80-DAG:  cvt.rn.bf16.f32 [[R0:%rs[0-9]+]], [[FR0]]
; SM80-DAG:  cvt.rn.bf16.f32 [[R1:%rs[0-9]+]], [[FR1]]
; SM80-DAG:  mov.b32        [[R:%r[0-9]+]], {[[R0]], [[R1]]}
;
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;

define <2 x bfloat> @test_fadd_imm_0(<2 x bfloat> %a) #0 {
  %r = fadd <2 x bfloat> <bfloat 1.0, bfloat 2.0>, %a
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_fadd_imm_1(
; CHECK:      ld.param.b16    [[A:%rs[0-9]+]], [test_fadd_imm_1_param_0];
; SM90:       mov.b16         [[B:%rs[0-9]+]], 0x3F80;
; SM90:       add.rn.bf16     [[R:%rs[0-9]+]], [[A]], [[B]];

; SM80-DAG:   cvt.f32.bf16    [[FA:%f[0-9]+]], [[A]];
; SM80:       add.rn.f32      [[FR:%f[0-9]+]], [[FA]], 0f3F800000;
; SM80:       cvt.rn.bf16.f32 [[R:%rs[0-9]+]], [[FR]];

; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;

define bfloat @test_fadd_imm_1(bfloat %a) #0 {
  %r = fadd bfloat %a, 1.0
  ret bfloat %r
}

; CHECK-LABEL: test_fsubx2(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fsubx2_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fsubx2_param_1];
; SM90:       sub.rn.bf16x2   [[R:%r[0-9]+]], [[A]], [[B]];

; SM80-DAG:   mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]];
; SM80-DAG:   mov.b32         {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]];
; SM80-DAG:   cvt.f32.bf16    [[FA1:%f[0-9]+]], [[A1]];
; SM80-DAG:   cvt.f32.bf16    [[FA0:%f[0-9]+]], [[A0]];
; SM80-DAG:   cvt.f32.bf16    [[FB0:%f[0-9]+]], [[B0]];
; SM80-DAG:   cvt.f32.bf16    [[FB1:%f[0-9]+]], [[B1]];
; SM80-DAG:   sub.rn.f32      [[FR0:%f[0-9]+]], [[FA0]], [[FB0]];
; SM80-DAG:   sub.rn.f32      [[FR1:%f[0-9]+]], [[FA1]], [[FB1]];
; SM80-DAG:   cvt.rn.bf16.f32 [[R0:%rs[0-9]+]], [[FR0]];
; SM80-DAG:   cvt.rn.bf16.f32 [[R1:%rs[0-9]+]], [[FR1]];
; SM80:       mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]};

; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;

define <2 x bfloat> @test_fsubx2(<2 x bfloat> %a, <2 x bfloat> %b) #0 {
  %r = fsub <2 x bfloat> %a, %b
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_fmulx2(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fmulx2_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fmulx2_param_1];
; SM90:       mul.rn.bf16x2   [[R:%r[0-9]+]], [[A]], [[B]];

; SM80-DAG:   mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]];
; SM80-DAG:   mov.b32         {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]];
; SM80-DAG:   cvt.f32.bf16    [[FA1:%f[0-9]+]], [[A1]];
; SM80-DAG:   cvt.f32.bf16    [[FA0:%f[0-9]+]], [[A0]];
; SM80-DAG:   cvt.f32.bf16    [[FB0:%f[0-9]+]], [[B0]];
; SM80-DAG:   cvt.f32.bf16    [[FB1:%f[0-9]+]], [[B1]];
; SM80-DAG:   mul.rn.f32      [[FR0:%f[0-9]+]], [[FA0]], [[FB0]];
; SM80-DAG:   mul.rn.f32      [[FR1:%f[0-9]+]], [[FA1]], [[FB1]];
; SM80-DAG:   cvt.rn.bf16.f32 [[R0:%rs[0-9]+]], [[FR0]];
; SM80-DAG:   cvt.rn.bf16.f32 [[R1:%rs[0-9]+]], [[FR1]];
; SM80:       mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]};

; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;

define <2 x bfloat> @test_fmulx2(<2 x bfloat> %a, <2 x bfloat> %b) #0 {
  %r = fmul <2 x bfloat> %a, %b
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_fdiv(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fdiv_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fdiv_param_1];
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  mov.b32         {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-DAG:  cvt.f32.bf16     [[FA0:%f[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.f32.bf16     [[FA1:%f[0-9]+]], [[A1]];
; CHECK-DAG:  cvt.f32.bf16     [[FB0:%f[0-9]+]], [[B0]];
; CHECK-DAG:  cvt.f32.bf16     [[FB1:%f[0-9]+]], [[B1]];
; CHECK-DAG:  div.rn.f32      [[FR0:%f[0-9]+]], [[FA0]], [[FB0]];
; CHECK-DAG:  div.rn.f32      [[FR1:%f[0-9]+]], [[FA1]], [[FB1]];
; CHECK-DAG:  cvt.rn.bf16.f32  [[R0:%rs[0-9]+]], [[FR0]];
; CHECK-DAG:  cvt.rn.bf16.f32  [[R1:%rs[0-9]+]], [[FR1]];
; CHECK-NEXT: mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;

define <2 x bfloat> @test_fdiv(<2 x bfloat> %a, <2 x bfloat> %b) #0 {
  %r = fdiv <2 x bfloat> %a, %b
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_fneg(
; CHECK-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_fneg_param_0];

; CHECK-DAG:        xor.b32        [[IHH0:%r[0-9]+]], [[A]], -2147450880;
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[IHH0]];
; CHECK-NEXT: ret;
define <2 x bfloat> @test_fneg(<2 x bfloat> %a) #0 {
  %r = fneg <2 x bfloat> %a
  ret <2 x bfloat> %r
}

; CHECK-LABEL: .func test_ldst_v2bf16(
; CHECK-DAG:    ld.param.u64    %[[A:rd[0-9]+]], [test_ldst_v2bf16_param_0];
; CHECK-DAG:    ld.param.u64    %[[B:rd[0-9]+]], [test_ldst_v2bf16_param_1];
; CHECK-DAG:    ld.b32          [[E:%r[0-9]+]], [%[[A]]]
; CHECK-DAG:    st.b32          [%[[B]]], [[E]];
; CHECK:        ret;
define void @test_ldst_v2bf16(ptr %a, ptr %b) {
  %t1 = load <2 x bfloat>, ptr %a
  store <2 x bfloat> %t1, ptr %b, align 16
  ret void
}

; CHECK-LABEL: .func test_ldst_v3bf16(
; CHECK-DAG:    ld.param.u64    %[[A:rd[0-9]+]], [test_ldst_v3bf16_param_0];
; CHECK-DAG:    ld.param.u64    %[[B:rd[0-9]+]], [test_ldst_v3bf16_param_1];
; -- v3 is inconvenient to capture as it's lowered as ld.b64 + fair
;    number of bitshifting instructions that may change at llvm's whim.
;    So we only verify that we only issue correct number of writes using
;    correct offset, but not the values we write.
; CHECK-DAG:    ld.u64
; CHECK-DAG:    st.u32          [%[[B]]],
; CHECK-DAG:    st.b16          [%[[B]]+4],
; CHECK:        ret;
define void @test_ldst_v3bf16(ptr %a, ptr %b) {
  %t1 = load <3 x bfloat>, ptr %a
  store <3 x bfloat> %t1, ptr %b, align 16
  ret void
}

declare <2 x bfloat> @test_callee(<2 x bfloat> %a, <2 x bfloat> %b) #0

; CHECK-LABEL: test_call(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_call_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_call_param_1];
; CHECK:      {
; CHECK-DAG:  .param .align 4 .b8 param0[4];
; CHECK-DAG:  .param .align 4 .b8 param1[4];
; CHECK-DAG:  st.param.b32    [param0+0], [[A]];
; CHECK-DAG:  st.param.b32    [param1+0], [[B]];
; CHECK-DAG:  .param .align 4 .b8 retval0[4];
; CHECK:      call.uni (retval0),
; CHECK-NEXT:        test_callee,
; CHECK:      );
; CHECK-NEXT: ld.param.b32    [[R:%r[0-9]+]], [retval0+0];
; CHECK-NEXT: }
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;

define <2 x bfloat> @test_call(<2 x bfloat> %a, <2 x bfloat> %b) #0 {
  %r = call <2 x bfloat> @test_callee(<2 x bfloat> %a, <2 x bfloat> %b)
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_select(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_select_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_select_param_1];
; CHECK-DAG:  ld.param.u8     [[C:%rs[0-9]+]], [test_select_param_2]
; CHECK-DAG:  setp.eq.b16     [[PRED:%p[0-9]+]], %rs{{.*}}, 1;
; CHECK-NEXT: selp.b32        [[R:%r[0-9]+]], [[A]], [[B]], [[PRED]];
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;

define <2 x bfloat> @test_select(<2 x bfloat> %a, <2 x bfloat> %b, i1 zeroext %c) #0 {
  %r = select i1 %c, <2 x bfloat> %a, <2 x bfloat> %b
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_select_cc(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_select_cc_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_select_cc_param_1];
; CHECK-DAG:  ld.param.b32    [[C:%r[0-9]+]], [test_select_cc_param_2];
; CHECK-DAG:  ld.param.b32    [[D:%r[0-9]+]], [test_select_cc_param_3];
;
; SM90:  setp.neu.bf16x2  [[P0:%p[0-9]+]]|[[P1:%p[0-9]+]], [[C]], [[D]]
;
; SM80-DAG: mov.b32        {[[C0:%rs[0-9]+]], [[C1:%rs[0-9]+]]}, [[C]]
; SM80-DAG: mov.b32        {[[D0:%rs[0-9]+]], [[D1:%rs[0-9]+]]}, [[D]]
; SM80-DAG: cvt.f32.bf16 [[DF0:%f[0-9]+]], [[D0]];
; SM80-DAG: cvt.f32.bf16 [[CF0:%f[0-9]+]], [[C0]];
; SM80-DAG: cvt.f32.bf16 [[DF1:%f[0-9]+]], [[D1]];
; SM80-DAG: cvt.f32.bf16 [[CF1:%f[0-9]+]], [[C1]];
; SM80-DAG: setp.neu.f32    [[P0:%p[0-9]+]], [[CF0]], [[DF0]]
; SM80-DAG: setp.neu.f32    [[P1:%p[0-9]+]], [[CF1]], [[DF1]]
;
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  mov.b32         {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-DAG:  selp.b16        [[R0:%rs[0-9]+]], [[A0]], [[B0]], [[P0]];
; CHECK-DAG:  selp.b16        [[R1:%rs[0-9]+]], [[A1]], [[B1]], [[P1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;

define <2 x bfloat> @test_select_cc(<2 x bfloat> %a, <2 x bfloat> %b, <2 x bfloat> %c, <2 x bfloat> %d) #0 {
  %cc = fcmp une <2 x bfloat> %c, %d
  %r = select <2 x i1> %cc, <2 x bfloat> %a, <2 x bfloat> %b
  ret <2 x bfloat> %r
}


; CHECK-LABEL: test_select_cc_f32_bf16(
; CHECK-DAG:  ld.param.v2.f32    {[[A0:%f[0-9]+]], [[A1:%f[0-9]+]]}, [test_select_cc_f32_bf16_param_0];
; CHECK-DAG:  ld.param.b32    [[C:%r[0-9]+]], [test_select_cc_f32_bf16_param_2];
; CHECK-DAG:  ld.param.b32    [[D:%r[0-9]+]], [test_select_cc_f32_bf16_param_3];
; SM90:  setp.neu.bf16x2  [[P0:%p[0-9]+]]|[[P1:%p[0-9]+]], [[C]], [[D]]
; CHECK-DAG:  ld.param.v2.f32    {[[B0:%f[0-9]+]], [[B1:%f[0-9]+]]}, [test_select_cc_f32_bf16_param_1];

; SM80-DAG: mov.b32         {[[C0:%rs[0-9]+]], [[C1:%rs[0-9]+]]}, [[C]]
; SM80-DAG: mov.b32         {[[D0:%rs[0-9]+]], [[D1:%rs[0-9]+]]}, [[D]]
; SM80-DAG: cvt.f32.bf16 [[DF0:%f[0-9]+]], [[D0]];
; SM80-DAG: cvt.f32.bf16 [[CF0:%f[0-9]+]], [[C0]];
; SM80-DAG: cvt.f32.bf16 [[DF1:%f[0-9]+]], [[D1]];
; SM80-DAG: cvt.f32.bf16 [[CF1:%f[0-9]+]], [[C1]];
; SM80-DAG: setp.neu.f32    [[P0:%p[0-9]+]], [[CF0]], [[DF0]]
; SM80-DAG: setp.neu.f32    [[P1:%p[0-9]+]], [[CF1]], [[DF1]]
;
; CHECK-DAG: selp.f32        [[R0:%f[0-9]+]], [[A0]], [[B0]], [[P0]];
; CHECK-DAG: selp.f32        [[R1:%f[0-9]+]], [[A1]], [[B1]], [[P1]];
; CHECK-NEXT: st.param.v2.f32    [func_retval0+0], {[[R0]], [[R1]]};
; CHECK-NEXT: ret;
define <2 x float> @test_select_cc_f32_bf16(<2 x float> %a, <2 x float> %b,
                                           <2 x bfloat> %c, <2 x bfloat> %d) #0 {
  %cc = fcmp une <2 x bfloat> %c, %d
  %r = select <2 x i1> %cc, <2 x float> %a, <2 x float> %b
  ret <2 x float> %r
}

; CHECK-LABEL: test_select_cc_bf16_f32(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_select_cc_bf16_f32_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_select_cc_bf16_f32_param_1];
; CHECK-DAG:  ld.param.v2.f32 {[[C0:%f[0-9]+]], [[C1:%f[0-9]+]]}, [test_select_cc_bf16_f32_param_2];
; CHECK-DAG:  ld.param.v2.f32 {[[D0:%f[0-9]+]], [[D1:%f[0-9]+]]}, [test_select_cc_bf16_f32_param_3];
; CHECK-DAG:  setp.neu.f32    [[P0:%p[0-9]+]], [[C0]], [[D0]]
; CHECK-DAG:  setp.neu.f32    [[P1:%p[0-9]+]], [[C1]], [[D1]]
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  mov.b32         {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-DAG:  selp.b16        [[R0:%rs[0-9]+]], [[A0]], [[B0]], [[P0]];
; CHECK-DAG:  selp.b16        [[R1:%rs[0-9]+]], [[A1]], [[B1]], [[P1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x bfloat> @test_select_cc_bf16_f32(<2 x bfloat> %a, <2 x bfloat> %b,
                                          <2 x float> %c, <2 x float> %d) #0 {
  %cc = fcmp une <2 x float> %c, %d
  %r = select <2 x i1> %cc, <2 x bfloat> %a, <2 x bfloat> %b
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_fptrunc_2xfloat(
; CHECK:      ld.param.v2.f32 {[[A0:%f[0-9]+]], [[A1:%f[0-9]+]]}, [test_fptrunc_2xfloat_param_0];
; CHECK-DAG:  cvt.rn.bf16.f32  [[R0:%rs[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.rn.bf16.f32  [[R1:%rs[0-9]+]], [[A1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x bfloat> @test_fptrunc_2xfloat(<2 x float> %a) #0 {
  %r = fptrunc <2 x float> %a to <2 x bfloat>
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_fpext_2xfloat(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_fpext_2xfloat_param_0];
; CHECK:      mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  cvt.f32.bf16     [[R0:%f[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.f32.bf16     [[R1:%f[0-9]+]], [[A1]];
; CHECK-NEXT: st.param.v2.f32 [func_retval0+0], {[[R0]], [[R1]]};
; CHECK:      ret;
define <2 x float> @test_fpext_2xfloat(<2 x bfloat> %a) #0 {
  %r = fpext <2 x bfloat> %a to <2 x float>
  ret <2 x float> %r
}

; CHECK-LABEL: test_bitcast_2xbf16_to_2xi16(
; CHECK:      ld.param.u32    [[A:%r[0-9]+]], [test_bitcast_2xbf16_to_2xi16_param_0];
; CHECK:      st.param.b32 [func_retval0+0], [[A]]
; CHECK:      ret;
define <2 x i16> @test_bitcast_2xbf16_to_2xi16(<2 x bfloat> %a) #0 {
  %r = bitcast <2 x bfloat> %a to <2 x i16>
  ret <2 x i16> %r
}


; CHECK-LABEL: test_bitcast_2xi16_to_2xbf16(
; CHECK:      ld.param.b32     [[R]], [test_bitcast_2xi16_to_2xbf16_param_0];
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x bfloat> @test_bitcast_2xi16_to_2xbf16(<2 x i16> %a) #0 {
  %r = bitcast <2 x i16> %a to <2 x bfloat>
  ret <2 x bfloat> %r
}

declare <2 x bfloat> @llvm.sqrt.f16(<2 x bfloat> %a) #0
declare <2 x bfloat> @llvm.powi.f16(<2 x bfloat> %a, <2 x i32> %b) #0
declare <2 x bfloat> @llvm.sin.f16(<2 x bfloat> %a) #0
declare <2 x bfloat> @llvm.cos.f16(<2 x bfloat> %a) #0
declare <2 x bfloat> @llvm.pow.f16(<2 x bfloat> %a, <2 x bfloat> %b) #0
declare <2 x bfloat> @llvm.exp.f16(<2 x bfloat> %a) #0
declare <2 x bfloat> @llvm.exp2.f16(<2 x bfloat> %a) #0
declare <2 x bfloat> @llvm.log.f16(<2 x bfloat> %a) #0
declare <2 x bfloat> @llvm.log10.f16(<2 x bfloat> %a) #0
declare <2 x bfloat> @llvm.log2.f16(<2 x bfloat> %a) #0
declare <2 x bfloat> @llvm.fma.f16(<2 x bfloat> %a, <2 x bfloat> %b, <2 x bfloat> %c) #0
declare <2 x bfloat> @llvm.fabs.f16(<2 x bfloat> %a) #0
declare <2 x bfloat> @llvm.minnum.f16(<2 x bfloat> %a, <2 x bfloat> %b) #0
declare <2 x bfloat> @llvm.maxnum.f16(<2 x bfloat> %a, <2 x bfloat> %b) #0
declare <2 x bfloat> @llvm.copysign.f16(<2 x bfloat> %a, <2 x bfloat> %b) #0
declare <2 x bfloat> @llvm.floor.f16(<2 x bfloat> %a) #0
declare <2 x bfloat> @llvm.ceil.f16(<2 x bfloat> %a) #0
declare <2 x bfloat> @llvm.trunc.f16(<2 x bfloat> %a) #0
declare <2 x bfloat> @llvm.rint.f16(<2 x bfloat> %a) #0
declare <2 x bfloat> @llvm.nearbyint.f16(<2 x bfloat> %a) #0
declare <2 x bfloat> @llvm.round.f16(<2 x bfloat> %a) #0
declare <2 x bfloat> @llvm.fmuladd.f16(<2 x bfloat> %a, <2 x bfloat> %b, <2 x bfloat> %c) #0


; CHECK-LABEL: test_sqrt(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_sqrt_param_0];
; CHECK:      mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  cvt.f32.bf16     [[AF0:%f[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.f32.bf16     [[AF1:%f[0-9]+]], [[A1]];
; CHECK-DAG:  sqrt.rn.f32     [[RF0:%f[0-9]+]], [[AF0]];
; CHECK-DAG:  sqrt.rn.f32     [[RF1:%f[0-9]+]], [[AF1]];
; CHECK-DAG:  cvt.rn.bf16.f32  [[R0:%rs[0-9]+]], [[RF0]];
; CHECK-DAG:  cvt.rn.bf16.f32  [[R1:%rs[0-9]+]], [[RF1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x bfloat> @test_sqrt(<2 x bfloat> %a) #0 {
  %r = call <2 x bfloat> @llvm.sqrt.f16(<2 x bfloat> %a)
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_fmuladd(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fmuladd_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fmuladd_param_1];
; CHECK-DAG:  ld.param.b32    [[C:%r[0-9]+]], [test_fmuladd_param_2];
;
; CHECK:       fma.rn.bf16x2   [[RA:%r[0-9]+]], [[A]], [[B]], [[C]];
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[RA]];
; CHECK:      ret;
define <2 x bfloat> @test_fmuladd(<2 x bfloat> %a, <2 x bfloat> %b, <2 x bfloat> %c) #0 {
  %r = call <2 x bfloat> @llvm.fmuladd.f16(<2 x bfloat> %a, <2 x bfloat> %b, <2 x bfloat> %c)
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_fabs(
; CHECK:      ld.param.u32    [[A:%r[0-9]+]], [test_fabs_param_0];
; CHECK:      and.b32         [[R:%r[0-9]+]], [[A]], 2147450879;
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x bfloat> @test_fabs(<2 x bfloat> %a) #0 {
  %r = call <2 x bfloat> @llvm.fabs.f16(<2 x bfloat> %a)
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_fabs_add(
; CHECK:      abs.bf16x2
; CHECK:      ret;
define <2 x bfloat> @test_fabs_add(<2 x bfloat> %a, <2 x bfloat> %b) #0 {
  %s = fadd <2 x bfloat> %a, %a
  %r = call <2 x bfloat> @llvm.fabs.f16(<2 x bfloat> %s)
  %d = fadd <2 x bfloat> %r, %b
  ret <2 x bfloat> %d
}


; CHECK-LABEL: test_minnum(
; CHECK-DAG:  ld.param.b32    [[AF0:%r[0-9]+]], [test_minnum_param_0];
; CHECK-DAG:  ld.param.b32    [[BF0:%r[0-9]+]], [test_minnum_param_1];
; CHECK-DAG:  min.bf16x2         [[RF0:%r[0-9]+]], [[AF0]], [[BF0]];
; CHECK:      st.param.b32    [func_retval0+0], [[RF0]];
; CHECK:      ret;
define <2 x bfloat> @test_minnum(<2 x bfloat> %a, <2 x bfloat> %b) #0 {
  %r = call <2 x bfloat> @llvm.minnum.f16(<2 x bfloat> %a, <2 x bfloat> %b)
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_maxnum(
; CHECK-DAG:  ld.param.b32    [[AF0:%r[0-9]+]], [test_maxnum_param_0];
; CHECK-DAG:  ld.param.b32    [[BF0:%r[0-9]+]], [test_maxnum_param_1];
; CHECK-DAG:  max.bf16x2         [[RF0:%r[0-9]+]], [[AF0]], [[BF0]];
; CHECK:      st.param.b32    [func_retval0+0], [[RF0]];
; CHECK:      ret;
define <2 x bfloat> @test_maxnum(<2 x bfloat> %a, <2 x bfloat> %b) #0 {
  %r = call <2 x bfloat> @llvm.maxnum.f16(<2 x bfloat> %a, <2 x bfloat> %b)
  ret <2 x bfloat> %r
}



; CHECK-LABEL: test_floor(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_floor_param_0];
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]];
; SM90:  cvt.rmi.bf16.bf16 [[R1:%rs[0-9]+]], [[A1]];
; SM90:  cvt.rmi.bf16.bf16 [[R0:%rs[0-9]+]], [[A0]];
; SM80-DAG:   cvt.f32.bf16     [[FA0:%f[0-9]+]], [[A0]];
; SM80-DAG:   cvt.f32.bf16     [[FA1:%f[0-9]+]], [[A1]];
; SM80-DAG:  cvt.rmi.f32.f32 [[RF0:%f[0-9]+]], [[FA0]];
; SM80-DAG:  cvt.rmi.f32.f32 [[RF1:%f[0-9]+]], [[FA1]];
; SM80-DAG:  cvt.rn.bf16.f32  [[R0:%rs[0-9]+]], [[RF0]];
; SM80-DAG:  cvt.rn.bf16.f32  [[R1:%rs[0-9]+]], [[RF1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x bfloat> @test_floor(<2 x bfloat> %a) #0 {
  %r = call <2 x bfloat> @llvm.floor.f16(<2 x bfloat> %a)
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_ceil(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_ceil_param_0];
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]];
; SM90:  cvt.rpi.bf16.bf16 [[R1:%rs[0-9]+]], [[A1]];
; SM90:  cvt.rpi.bf16.bf16 [[R0:%rs[0-9]+]], [[A0]];
; SM80-DAG:   cvt.f32.bf16     [[FA0:%f[0-9]+]], [[A0]];
; SM80-DAG:   cvt.f32.bf16     [[FA1:%f[0-9]+]], [[A1]];
; SM80-DAG:   cvt.rpi.f32.f32 [[RF0:%f[0-9]+]], [[FA0]];
; SM80-DAG:   cvt.rpi.f32.f32 [[RF1:%f[0-9]+]], [[FA1]];
; SM80-DAG:  cvt.rn.bf16.f32  [[R0:%rs[0-9]+]], [[RF0]];
; SM80-DAG:  cvt.rn.bf16.f32  [[R1:%rs[0-9]+]], [[RF1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x bfloat> @test_ceil(<2 x bfloat> %a) #0 {
  %r = call <2 x bfloat> @llvm.ceil.f16(<2 x bfloat> %a)
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_trunc(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_trunc_param_0];
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]];
; SM90:  cvt.rzi.bf16.bf16 [[R1:%rs[0-9]+]], [[A1]];
; SM90:  cvt.rzi.bf16.bf16 [[R0:%rs[0-9]+]], [[A0]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x bfloat> @test_trunc(<2 x bfloat> %a) #0 {
  %r = call <2 x bfloat> @llvm.trunc.f16(<2 x bfloat> %a)
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_rint(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_rint_param_0];
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]];
; SM90:  cvt.rni.bf16.bf16 [[R1:%rs[0-9]+]], [[A1]];
; SM90:  cvt.rni.bf16.bf16 [[R0:%rs[0-9]+]], [[A0]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x bfloat> @test_rint(<2 x bfloat> %a) #0 {
  %r = call <2 x bfloat> @llvm.rint.f16(<2 x bfloat> %a)
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_round(
; CHECK:      ld.param.b32    {{.*}}, [test_round_param_0];
; check the use of sign mask and 0.5 to implement round
; CHECK:      and.b32 [[R1:%r[0-9]+]], {{.*}}, -2147483648;
; CHECK:      or.b32 {{.*}}, [[R1]], 1056964608;
; CHECK:      and.b32 [[R2:%r[0-9]+]], {{.*}}, -2147483648;
; CHECK:      or.b32 {{.*}}, [[R2]], 1056964608;
; CHECK:      st.param.b32    [func_retval0+0], {{.*}};
; CHECK:      ret;
define <2 x bfloat> @test_round(<2 x bfloat> %a) #0 {
  %r = call <2 x bfloat> @llvm.round.f16(<2 x bfloat> %a)
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_copysign(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_copysign_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_copysign_param_1];
; SM80-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; SM80-DAG:  mov.b32         {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; SM80-DAG:  abs.bf16        [[AW1:%rs[0-9]+]], [[A1]];
; SM80-DAG:  neg.bf16        [[AY1:%rs[0-9]+]], [[AW1]];
; SM80-DAG:  shr.u16         [[BS1:%rs[0-9]+]], [[B1]], 15;
; SM80-DAG:  and.b16         [[BR1:%rs[0-9]+]], [[BS1]], 1;
; SM80-DAG:  setp.eq.b16     [[P1:%p[0-9]+]], [[BR1]], 1;
; SM80-DAG:  selp.b16        [[RS1:%rs[0-9]+]], [[AY1]], [[AW1]], [[P1]]
; SM80-DAG:  abs.bf16        [[AW0:%rs[0-9]+]], [[A0]];
; SM80-DAG:  neg.bf16        [[AY0:%rs[0-9]+]], [[AW0]];
; SM80-DAG:  shr.u16         [[BS0:%rs[0-9]+]], [[B0]], 15;
; SM80-DAG:  and.b16         [[BR0:%rs[0-9]+]], [[BS0]], 1;
; SM80-DAG:  setp.eq.b16     [[P0:%p[0-9]+]], [[BR0]], 1;
; SM80-DAG:  selp.b16        [[RS0:%rs[0-9]+]], [[AY0]], [[AW0]], [[P0]]
; SM80-DAG:  mov.b32         [[R:%r[0-9]+]], {[[RS0]], [[RS1]]}
; SM90-DAG:  and.b32         [[R1:%r[0-9]+]], [[B]], -2147450880;
; SM90-DAG:  and.b32         [[R2:%r[0-9]+]], [[A]], 2147450879;
; SM90-DAG:  or.b32          [[R:%r[0-9]+]], [[R2]], [[R1]];
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x bfloat> @test_copysign(<2 x bfloat> %a, <2 x bfloat> %b) #0 {
  %r = call <2 x bfloat> @llvm.copysign.f16(<2 x bfloat> %a, <2 x bfloat> %b)
  ret <2 x bfloat> %r
}

