; RUN: llc < %s -march=nvptx64 -mcpu=sm_80 -mattr=+ptx70 | FileCheck --check-prefixes=CHECK,SM80 %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_90 -mattr=+ptx78 | FileCheck --check-prefixes=CHECK,SM90 %s
; RUN: %if ptxas-11.8 %{ llc < %s -march=nvptx64 -mcpu=sm_80 -mattr=+ptx71 | %ptxas-verify -arch=sm_80 %}
; RUN: %if ptxas-11.8 %{ llc < %s -march=nvptx64 -mcpu=sm_90 -mattr=+ptx78 | %ptxas-verify -arch=sm_90 %}

; LDST: .b8 bfloat_array[8] = {1, 2, 3, 4, 5, 6, 7, 8};
@"bfloat_array" = addrspace(1) constant [4 x bfloat]
                [bfloat 0xR0201, bfloat 0xR0403, bfloat 0xR0605, bfloat 0xR0807]

; CHECK-LABEL: test_fadd(
; CHECK-DAG:  ld.param.b16    [[A:%rs[0-9]+]], [test_fadd_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%rs[0-9]+]], [test_fadd_param_1];
; SM90:       add.rn.bf16     [[R:%rs[0-9]+]], [[A]], [[B]];
;
; SM80-DAG:   cvt.f32.bf16    [[FA:%f[0-9]+]], [[A]];
; SM80-DAG:   cvt.f32.bf16    [[FB:%f[0-9]+]], [[B]];
; SM80:       add.rn.f32      [[FR:%f[0-9]+]], [[FA]], [[FB]];
; SM80:       cvt.rn.bf16.f32 [[R:%rs[0-9]+]], [[FR]];
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;

define bfloat @test_fadd(bfloat %0, bfloat %1) {
  %3 = fadd bfloat %0, %1
  ret bfloat %3
}

; CHECK-LABEL: test_fsub(
; CHECK-DAG:  ld.param.b16    [[A:%rs[0-9]+]], [test_fsub_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%rs[0-9]+]], [test_fsub_param_1];
; SM90:       sub.rn.bf16     [[R:%rs[0-9]+]], [[A]], [[B]];
;
; SM80-DAG:   cvt.f32.bf16    [[FA:%f[0-9]+]], [[A]];
; SM80-DAG:   cvt.f32.bf16    [[FB:%f[0-9]+]], [[B]];
; SM80:       sub.rn.f32      [[FR:%f[0-9]+]], [[FA]], [[FB]];
; SM80:       cvt.rn.bf16.f32 [[R:%rs[0-9]+]], [[FR]];
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;

define bfloat @test_fsub(bfloat %0, bfloat %1) {
  %3 = fsub bfloat %0, %1
  ret bfloat %3
}

; CHECK-LABEL: test_faddx2(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_faddx2_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_faddx2_param_1];
; SM90:       add.rn.bf16x2   [[R:%r[0-9]+]], [[A]], [[B]];

; SM80-DAG:   mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]];
; SM80-DAG:   mov.b32         {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]];
; SM80-DAG:   cvt.f32.bf16    [[FA1:%f[0-9]+]], [[A1]];
; SM80-DAG:   cvt.f32.bf16    [[FA0:%f[0-9]+]], [[A0]];
; SM80-DAG:   cvt.f32.bf16    [[FB0:%f[0-9]+]], [[B0]];
; SM80-DAG:   cvt.f32.bf16    [[FB1:%f[0-9]+]], [[B1]];
; SM80-DAG:   add.rn.f32      [[FR0:%f[0-9]+]], [[FA0]], [[FB0]];
; SM80-DAG:   add.rn.f32      [[FR1:%f[0-9]+]], [[FA1]], [[FB1]];
; SM80-DAG:   cvt.rn.bf16.f32 [[R0:%rs[0-9]+]], [[FR0]];
; SM80-DAG:   cvt.rn.bf16.f32 [[R1:%rs[0-9]+]], [[FR1]];
; SM80:       mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]};
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;

define <2 x bfloat> @test_faddx2(<2 x bfloat> %a, <2 x bfloat> %b) #0 {
  %r = fadd <2 x bfloat> %a, %b
  ret <2 x bfloat> %r
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

; CHECK-LABEL: test_extract_0(
; CHECK:      ld.param.b16    [[A:%rs[0-9]+]], [test_extract_0_param_0];
; CHECK:      st.param.b16    [func_retval0+0], [[A]];
; CHECK:      ret;

define bfloat @test_extract_0(<2 x bfloat> %a) #0 {
  %e = extractelement <2 x bfloat> %a, i32 0
  ret bfloat %e
}

; CHECK-LABEL: test_extract_1(
; CHECK:      ld.param.b16    [[A:%rs[0-9]+]], [test_extract_1_param_0+2];
; CHECK:      st.param.b16    [func_retval0+0], [[A]];
; CHECK:      ret;

define bfloat @test_extract_1(<2 x bfloat> %a) #0 {
  %e = extractelement <2 x bfloat> %a, i32 1
  ret bfloat %e
}

; CHECK-LABEL: test_fpext_float(
; CHECK:      ld.param.b16    [[A:%rs[0-9]+]], [test_fpext_float_param_0];
; CHECK:      cvt.f32.bf16     [[R:%f[0-9]+]], [[A]];
; CHECK:      st.param.f32    [func_retval0+0], [[R]];
; CHECK:      ret;
define float @test_fpext_float(bfloat %a) #0 {
  %r = fpext bfloat %a to float
  ret float %r
}

; CHECK-LABEL: test_fptrunc_float(
; CHECK:      ld.param.f32    [[A:%f[0-9]+]], [test_fptrunc_float_param_0];
; CHECK:      cvt.rn.bf16.f32  [[R:%rs[0-9]+]], [[A]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define bfloat @test_fptrunc_float(float %a) #0 {
  %r = fptrunc float %a to bfloat
  ret bfloat %r
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

; CHECK-LABEL: test_select_cc_bf16_f64(
; CHECK-DAG:      ld.param.f64    [[A:%fd[0-9]+]], [test_select_cc_bf16_f64_param_0];
; CHECK-DAG:      ld.param.f64    [[B:%fd[0-9]+]], [test_select_cc_bf16_f64_param_1];
; CHECK:          setp.lt.f64     [[P:%p[0-9]+]], [[A]], [[B]];
; CHECK-DAG:      ld.param.b16    [[C:%rs[0-9]+]], [test_select_cc_bf16_f64_param_2];
; CHECK-DAG:      ld.param.b16    [[D:%rs[0-9]+]], [test_select_cc_bf16_f64_param_3];
; CHECK:          selp.b16        [[R:%rs[0-9]+]], [[C]], [[D]], [[P]];
; CHECK-NEXT:     st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT:     ret;
define bfloat @test_select_cc_bf16_f64(double %a, double %b, bfloat %c, bfloat %d) #0 {
  %cc = fcmp olt double %a, %b
  %r = select i1 %cc, bfloat %c, bfloat %d
  ret bfloat %r
}
