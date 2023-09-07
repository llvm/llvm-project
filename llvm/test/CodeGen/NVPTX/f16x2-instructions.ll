; ## Full FP16 support enabled by default.
; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_53 -asm-verbose=false \
; RUN:          -O0 -disable-post-ra -frame-pointer=all -verify-machineinstrs \
; RUN: | FileCheck -allow-deprecated-dag-overlap -check-prefixes CHECK,CHECK-F16 %s
; RUN: %if ptxas %{                                                           \
; RUN:   llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_53 -asm-verbose=false \
; RUN:          -O0 -disable-post-ra -frame-pointer=all -verify-machineinstrs \
; RUN:   | %ptxas-verify -arch=sm_53                                          \
; RUN: %}
; ## FP16 support explicitly disabled.
; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_53 -asm-verbose=false \
; RUN:          -O0 -disable-post-ra -frame-pointer=all --nvptx-no-f16-math \
; RUN:           -verify-machineinstrs \
; RUN: | FileCheck -allow-deprecated-dag-overlap -check-prefixes CHECK,CHECK-NOF16 %s
; RUN: %if ptxas %{                                                           \
; RUN:   llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_53 -asm-verbose=false \
; RUN:          -O0 -disable-post-ra -frame-pointer=all --nvptx-no-f16-math   \
; RUN:           -verify-machineinstrs                                        \
; RUN:   | %ptxas-verify -arch=sm_53                                          \
; RUN: %}
; ## FP16 is not supported by hardware.
; RUN: llc < %s -O0 -mtriple=nvptx64-nvidia-cuda -mcpu=sm_52 -asm-verbose=false \
; RUN:          -disable-post-ra -frame-pointer=all -verify-machineinstrs \
; RUN: | FileCheck -allow-deprecated-dag-overlap -check-prefixes CHECK,CHECK-NOF16 %s
; RUN: %if ptxas %{                                                               \
; RUN:   llc < %s -O0 -mtriple=nvptx64-nvidia-cuda -mcpu=sm_52 -asm-verbose=false \
; RUN:          -disable-post-ra -frame-pointer=all -verify-machineinstrs         \
; RUN:   | %ptxas-verify -arch=sm_52                                              \
; RUN: %}

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: test_ret_const(
; CHECK:     mov.b32         [[R:%r[0-9+]]], 1073757184;
; CHECK:     st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x half> @test_ret_const() #0 {
  ret <2 x half> <half 1.0, half 2.0>
}

; CHECK-LABEL: test_extract_0(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_extract_0_param_0];
; CHECK:      mov.b32         {[[R:%rs[0-9]+]], tmp}, [[A]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_extract_0(<2 x half> %a) #0 {
  %e = extractelement <2 x half> %a, i32 0
  ret half %e
}

; CHECK-LABEL: test_extract_1(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_extract_1_param_0];
; CHECK:      mov.b32         {tmp, [[R:%rs[0-9]+]]}, [[A]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_extract_1(<2 x half> %a) #0 {
  %e = extractelement <2 x half> %a, i32 1
  ret half %e
}

; CHECK-LABEL: test_extract_i(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_extract_i_param_0];
; CHECK-DAG:  ld.param.u64    [[IDX:%rd[0-9]+]], [test_extract_i_param_1];
; CHECK-DAG:  setp.eq.s64     [[PRED:%p[0-9]+]], [[IDX]], 0;
; CHECK-DAG:  mov.b32         {[[E0:%rs[0-9]+]], [[E1:%rs[0-9]+]]}, [[A]];
; CHECK:      selp.b16        [[R:%rs[0-9]+]], [[E0]], [[E1]], [[PRED]];
; CHECK:      st.param.b16    [func_retval0+0], [[R]];
; CHECK:      ret;
define half @test_extract_i(<2 x half> %a, i64 %idx) #0 {
  %e = extractelement <2 x half> %a, i64 %idx
  ret half %e
}

; CHECK-LABEL: test_fadd(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fadd_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fadd_param_1];
;
; CHECK-F16-NEXT:   add.rn.f16x2   [[R:%r[0-9]+]], [[A]], [[B]];
;
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  add.rn.f32     [[FR0:%f[0-9]+]], [[FA0]], [[FB0]];
; CHECK-NOF16-DAG:  add.rn.f32     [[FR1:%f[0-9]+]], [[FA1]], [[FB1]];
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R0:%rs[0-9]+]], [[FR0]]
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R1:%rs[0-9]+]], [[FR1]]
; CHECK-NOF16:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
;
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x half> @test_fadd(<2 x half> %a, <2 x half> %b) #0 {
  %r = fadd <2 x half> %a, %b
  ret <2 x half> %r
}

; Check that we can lower fadd with immediate arguments.
; CHECK-LABEL: test_fadd_imm_0(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fadd_imm_0_param_0];
;
; CHECK-F16:        mov.b32        [[I:%r[0-9+]]], 1073757184;
; CHECK-F16:        add.rn.f16x2   [[R:%r[0-9]+]], [[A]], [[I]];
;
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  add.rn.f32     [[FR0:%f[0-9]+]], [[FA0]], 0f3F800000;
; CHECK-NOF16-DAG:  add.rn.f32     [[FR1:%f[0-9]+]], [[FA1]], 0f40000000;
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R0:%rs[0-9]+]], [[FR0]]
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R1:%rs[0-9]+]], [[FR1]]
; CHECK-NOF16:      mov.b32        [[R:%r[0-9]+]], {[[R0]], [[R1]]}
;
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x half> @test_fadd_imm_0(<2 x half> %a) #0 {
  %r = fadd <2 x half> <half 1.0, half 2.0>, %a
  ret <2 x half> %r
}

; CHECK-LABEL: test_fadd_imm_1(
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fadd_imm_1_param_0];
;
; CHECK-F16:        mov.b32        [[I:%r[0-9+]]], 1073757184;
; CHECK-F16:        add.rn.f16x2   [[R:%r[0-9]+]], [[B]], [[I]];
;
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  add.rn.f32     [[FR0:%f[0-9]+]], [[FA0]], 0f3F800000;
; CHECK-NOF16-DAG:  add.rn.f32     [[FR1:%f[0-9]+]], [[FA1]], 0f40000000;
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R0:%rs[0-9]+]], [[FR0]]
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R1:%rs[0-9]+]], [[FR1]]
; CHECK-NOF16:      mov.b32        [[R:%r[0-9]+]], {[[R0]], [[R1]]}
;
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x half> @test_fadd_imm_1(<2 x half> %a) #0 {
  %r = fadd <2 x half> %a, <half 1.0, half 2.0>
  ret <2 x half> %r
}

; CHECK-LABEL: test_fsub(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fsub_param_0];
;
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fsub_param_1];
; CHECK-F16-NEXT:   sub.rn.f16x2   [[R:%r[0-9]+]], [[A]], [[B]];
;
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  sub.rn.f32     [[FR0:%f[0-9]+]], [[FA0]], [[FB0]];
; CHECK-NOF16-DAG:  sub.rn.f32     [[FR1:%f[0-9]+]], [[FA1]], [[FB1]];
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R0:%rs[0-9]+]], [[FR0]]
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R1:%rs[0-9]+]], [[FR1]]
; CHECK-NOF16:      mov.b32        [[R:%r[0-9]+]], {[[R0]], [[R1]]}
;
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x half> @test_fsub(<2 x half> %a, <2 x half> %b) #0 {
  %r = fsub <2 x half> %a, %b
  ret <2 x half> %r
}

; CHECK-LABEL: test_fneg(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fneg_param_0];
;
; CHECK-F16:        mov.b32        [[I:%r[0-9+]]], 0;
; CHECK-F16-NEXT:   sub.rn.f16x2   [[R:%r[0-9]+]], [[I]], [[A]];
;
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  mov.f32        [[Z:%f[0-9]+]], 0f00000000;
; CHECK-NOF16-DAG:  sub.rn.f32     [[FR0:%f[0-9]+]], [[Z]], [[FA0]];
; CHECK-NOF16-DAG:  sub.rn.f32     [[FR1:%f[0-9]+]], [[Z]], [[FA1]];
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R0:%rs[0-9]+]], [[FR0]]
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R1:%rs[0-9]+]], [[FR1]]
; CHECK-NOF16:      mov.b32        [[R:%r[0-9]+]], {[[R0]], [[R1]]}
;
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x half> @test_fneg(<2 x half> %a) #0 {
  %r = fsub <2 x half> <half 0.0, half 0.0>, %a
  ret <2 x half> %r
}

; CHECK-LABEL: test_fmul(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fmul_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fmul_param_1];
; CHECK-F16-NEXT: mul.rn.f16x2     [[R:%r[0-9]+]], [[A]], [[B]];
;
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  mul.rn.f32     [[FR0:%f[0-9]+]], [[FA0]], [[FB0]];
; CHECK-NOF16-DAG:  mul.rn.f32     [[FR1:%f[0-9]+]], [[FA1]], [[FB1]];
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R0:%rs[0-9]+]], [[FR0]]
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R1:%rs[0-9]+]], [[FR1]]
; CHECK-NOF16:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
;
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x half> @test_fmul(<2 x half> %a, <2 x half> %b) #0 {
  %r = fmul <2 x half> %a, %b
  ret <2 x half> %r
}

; CHECK-LABEL: test_fdiv(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fdiv_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fdiv_param_1];
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  mov.b32         {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-DAG:  cvt.f32.f16     [[FA0:%f[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.f32.f16     [[FA1:%f[0-9]+]], [[A1]];
; CHECK-DAG:  cvt.f32.f16     [[FB0:%f[0-9]+]], [[B0]];
; CHECK-DAG:  cvt.f32.f16     [[FB1:%f[0-9]+]], [[B1]];
; CHECK-DAG:  div.rn.f32      [[FR0:%f[0-9]+]], [[FA0]], [[FB0]];
; CHECK-DAG:  div.rn.f32      [[FR1:%f[0-9]+]], [[FA1]], [[FB1]];
; CHECK-DAG:  cvt.rn.f16.f32  [[R0:%rs[0-9]+]], [[FR0]];
; CHECK-DAG:  cvt.rn.f16.f32  [[R1:%rs[0-9]+]], [[FR1]];
; CHECK-NEXT: mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x half> @test_fdiv(<2 x half> %a, <2 x half> %b) #0 {
  %r = fdiv <2 x half> %a, %b
  ret <2 x half> %r
}

; CHECK-LABEL: test_frem(
; -- Load two 16x2 inputs and split them into f16 elements
; CHECK-DAG:  ld.param.b32       [[A:%r[0-9]+]], [test_frem_param_0];
; CHECK-DAG:  ld.param.b32       [[B:%r[0-9]+]], [test_frem_param_1];
; -- Split into elements
; CHECK-DAG:  mov.b32            {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  mov.b32            {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; -- promote to f32.
; CHECK-DAG:  cvt.f32.f16        [[FA0:%f[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.f32.f16        [[FB0:%f[0-9]+]], [[B0]];
; CHECK-DAG:  cvt.f32.f16        [[FA1:%f[0-9]+]], [[A1]];
; CHECK-DAG:  cvt.f32.f16        [[FB1:%f[0-9]+]], [[B1]];
; -- frem(a[0],b[0]).
; CHECK-DAG:  div.rn.f32         [[FD0:%f[0-9]+]], [[FA0]], [[FB0]];
; CHECK-DAG:  cvt.rzi.f32.f32    [[DI0:%f[0-9]+]], [[FD0]];
; CHECK-DAG:  mul.f32            [[RI0:%f[0-9]+]], [[DI0]], [[FB0]];
; CHECK-DAG:  sub.f32            [[RFNINF0:%f[0-9]+]], [[FA0]], [[RI0]];
; CHECK-DAG:  testp.infinite.f32 [[ISB0INF:%p[0-9]+]], [[FB0]];
; CHECK-DAG:  selp.f32           [[RF0:%f[0-9]+]], [[FA0]], [[RFNINF0]], [[ISB0INF]];
; -- frem(a[1],b[1]).
; CHECK-DAG:  div.rn.f32         [[FD1:%f[0-9]+]], [[FA1]], [[FB1]];
; CHECK-DAG:  cvt.rzi.f32.f32    [[DI1:%f[0-9]+]], [[FD1]];
; CHECK-DAG:  mul.f32            [[RI1:%f[0-9]+]], [[DI1]], [[FB1]];
; CHECK-DAG:  sub.f32            [[RFNINF1:%f[0-9]+]], [[FA1]], [[RI1]];
; CHECK-DAG:  testp.infinite.f32 [[ISB1INF:%p[0-9]+]], [[FB1]];
; CHECK-DAG:  selp.f32           [[RF1:%f[0-9]+]], [[FA1]], [[RFNINF1]], [[ISB1INF]];
; -- convert back to f16.
; CHECK-DAG:  cvt.rn.f16.f32  [[R0:%rs[0-9]+]], [[RF0]];
; CHECK-DAG:  cvt.rn.f16.f32  [[R1:%rs[0-9]+]], [[RF1]];
; -- merge into f16x2 and return it.
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x half> @test_frem(<2 x half> %a, <2 x half> %b) #0 {
  %r = frem <2 x half> %a, %b
  ret <2 x half> %r
}

; CHECK-LABEL: .func test_ldst_v2f16(
; CHECK-DAG:    ld.param.u64    %[[A:rd[0-9]+]], [test_ldst_v2f16_param_0];
; CHECK-DAG:    ld.param.u64    %[[B:rd[0-9]+]], [test_ldst_v2f16_param_1];
; CHECK-DAG:    ld.b32          [[E:%r[0-9]+]], [%[[A]]]
; CHECK-DAG:    st.b32          [%[[B]]], [[E]];
; CHECK:        ret;
define void @test_ldst_v2f16(ptr %a, ptr %b) {
  %t1 = load <2 x half>, ptr %a
  store <2 x half> %t1, ptr %b, align 16
  ret void
}

; CHECK-LABEL: .func test_ldst_v3f16(
; CHECK-DAG:    ld.param.u64    %[[A:rd[0-9]+]], [test_ldst_v3f16_param_0];
; CHECK-DAG:    ld.param.u64    %[[B:rd[0-9]+]], [test_ldst_v3f16_param_1];
; -- v3 is inconvenient to capture as it's lowered as ld.b64 + fair
;    number of bitshifting instructions that may change at llvm's whim.
;    So we only verify that we only issue correct number of writes using
;    correct offset, but not the values we write.
; CHECK-DAG:    ld.u64
; CHECK-DAG:    st.u32          [%[[B]]],
; CHECK-DAG:    st.b16          [%[[B]]+4],
; CHECK:        ret;
define void @test_ldst_v3f16(ptr %a, ptr %b) {
  %t1 = load <3 x half>, ptr %a
  store <3 x half> %t1, ptr %b, align 16
  ret void
}

; CHECK-LABEL: .func test_ldst_v4f16(
; CHECK-DAG:    ld.param.u64    %[[A:rd[0-9]+]], [test_ldst_v4f16_param_0];
; CHECK-DAG:    ld.param.u64    %[[B:rd[0-9]+]], [test_ldst_v4f16_param_1];
; CHECK-DAG:    ld.v4.b16       {[[E0:%rs[0-9]+]], [[E1:%rs[0-9]+]], [[E2:%rs[0-9]+]], [[E3:%rs[0-9]+]]}, [%[[A]]];
; CHECK-DAG:    st.v4.b16       [%[[B]]], {[[E0]], [[E1]], [[E2]], [[E3]]};
; CHECK:        ret;
define void @test_ldst_v4f16(ptr %a, ptr %b) {
  %t1 = load <4 x half>, ptr %a
  store <4 x half> %t1, ptr %b, align 16
  ret void
}

; CHECK-LABEL: .func test_ldst_v8f16(
; CHECK-DAG:    ld.param.u64    %[[A:rd[0-9]+]], [test_ldst_v8f16_param_0];
; CHECK-DAG:    ld.param.u64    %[[B:rd[0-9]+]], [test_ldst_v8f16_param_1];
; CHECK-DAG:    ld.v4.b32       {[[E0:%r[0-9]+]], [[E1:%r[0-9]+]], [[E2:%r[0-9]+]], [[E3:%r[0-9]+]]}, [%[[A]]];
; CHECK-DAG:    st.v4.b32       [%[[B]]], {[[E0]], [[E1]], [[E2]], [[E3]]};
; CHECK:        ret;
define void @test_ldst_v8f16(ptr %a, ptr %b) {
  %t1 = load <8 x half>, ptr %a
  store <8 x half> %t1, ptr %b, align 16
  ret void
}

declare <2 x half> @test_callee(<2 x half> %a, <2 x half> %b) #0

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
define <2 x half> @test_call(<2 x half> %a, <2 x half> %b) #0 {
  %r = call <2 x half> @test_callee(<2 x half> %a, <2 x half> %b)
  ret <2 x half> %r
}

; CHECK-LABEL: test_call_flipped(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_call_flipped_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_call_flipped_param_1];
; CHECK:      {
; CHECK-DAG:  .param .align 4 .b8 param0[4];
; CHECK-DAG:  .param .align 4 .b8 param1[4];
; CHECK-DAG:  st.param.b32    [param0+0], [[B]];
; CHECK-DAG:  st.param.b32    [param1+0], [[A]];
; CHECK-DAG:  .param .align 4 .b8 retval0[4];
; CHECK:      call.uni (retval0),
; CHECK-NEXT:        test_callee,
; CHECK:      );
; CHECK-NEXT: ld.param.b32    [[R:%r[0-9]+]], [retval0+0];
; CHECK-NEXT: }
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x half> @test_call_flipped(<2 x half> %a, <2 x half> %b) #0 {
  %r = call <2 x half> @test_callee(<2 x half> %b, <2 x half> %a)
  ret <2 x half> %r
}

; CHECK-LABEL: test_tailcall_flipped(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_tailcall_flipped_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_tailcall_flipped_param_1];
; CHECK:      {
; CHECK-DAG:  .param .align 4 .b8 param0[4];
; CHECK-DAG:  .param .align 4 .b8 param1[4];
; CHECK-DAG:  st.param.b32    [param0+0], [[B]];
; CHECK-DAG:  st.param.b32    [param1+0], [[A]];
; CHECK-DAG:  .param .align 4 .b8 retval0[4];
; CHECK:      call.uni (retval0),
; CHECK-NEXT:        test_callee,
; CHECK:      );
; CHECK-NEXT: ld.param.b32    [[R:%r[0-9]+]], [retval0+0];
; CHECK-NEXT: }
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x half> @test_tailcall_flipped(<2 x half> %a, <2 x half> %b) #0 {
  %r = tail call <2 x half> @test_callee(<2 x half> %b, <2 x half> %a)
  ret <2 x half> %r
}

; CHECK-LABEL: test_select(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_select_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_select_param_1];
; CHECK-DAG:  ld.param.u8     [[C:%rs[0-9]+]], [test_select_param_2]
; CHECK-DAG:  setp.eq.b16     [[PRED:%p[0-9]+]], %rs{{.*}}, 1;
; CHECK-NEXT: selp.b32        [[R:%r[0-9]+]], [[A]], [[B]], [[PRED]];
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x half> @test_select(<2 x half> %a, <2 x half> %b, i1 zeroext %c) #0 {
  %r = select i1 %c, <2 x half> %a, <2 x half> %b
  ret <2 x half> %r
}

; CHECK-LABEL: test_select_cc(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_select_cc_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_select_cc_param_1];
; CHECK-DAG:  ld.param.b32    [[C:%r[0-9]+]], [test_select_cc_param_2];
; CHECK-DAG:  ld.param.b32    [[D:%r[0-9]+]], [test_select_cc_param_3];
;
; CHECK-F16:  setp.neu.f16x2  [[P0:%p[0-9]+]]|[[P1:%p[0-9]+]], [[C]], [[D]]
;
; CHECK-NOF16-DAG: mov.b32        {[[C0:%rs[0-9]+]], [[C1:%rs[0-9]+]]}, [[C]]
; CHECK-NOF16-DAG: mov.b32        {[[D0:%rs[0-9]+]], [[D1:%rs[0-9]+]]}, [[D]]
; CHECK-NOF16-DAG: cvt.f32.f16 [[DF0:%f[0-9]+]], [[D0]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[CF0:%f[0-9]+]], [[C0]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[DF1:%f[0-9]+]], [[D1]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[CF1:%f[0-9]+]], [[C1]];
; CHECK-NOF16-DAG: setp.neu.f32    [[P0:%p[0-9]+]], [[CF0]], [[DF0]]
; CHECK-NOF16-DAG: setp.neu.f32    [[P1:%p[0-9]+]], [[CF1]], [[DF1]]
;
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  mov.b32         {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-DAG:  selp.b16        [[R0:%rs[0-9]+]], [[A0]], [[B0]], [[P0]];
; CHECK-DAG:  selp.b16        [[R1:%rs[0-9]+]], [[A1]], [[B1]], [[P1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x half> @test_select_cc(<2 x half> %a, <2 x half> %b, <2 x half> %c, <2 x half> %d) #0 {
  %cc = fcmp une <2 x half> %c, %d
  %r = select <2 x i1> %cc, <2 x half> %a, <2 x half> %b
  ret <2 x half> %r
}

; CHECK-LABEL: test_select_cc_f32_f16(
; CHECK-DAG:  ld.param.v2.f32    {[[A0:%f[0-9]+]], [[A1:%f[0-9]+]]}, [test_select_cc_f32_f16_param_0];
; CHECK-DAG:  ld.param.v2.f32    {[[B0:%f[0-9]+]], [[B1:%f[0-9]+]]}, [test_select_cc_f32_f16_param_1];
; CHECK-DAG:  ld.param.b32    [[C:%r[0-9]+]], [test_select_cc_f32_f16_param_2];
; CHECK-DAG:  ld.param.b32    [[D:%r[0-9]+]], [test_select_cc_f32_f16_param_3];
;
; CHECK-F16:  setp.neu.f16x2  [[P0:%p[0-9]+]]|[[P1:%p[0-9]+]], [[C]], [[D]]
; CHECK-NOF16-DAG: mov.b32         {[[C0:%rs[0-9]+]], [[C1:%rs[0-9]+]]}, [[C]]
; CHECK-NOF16-DAG: mov.b32         {[[D0:%rs[0-9]+]], [[D1:%rs[0-9]+]]}, [[D]]
; CHECK-NOF16-DAG: cvt.f32.f16 [[DF0:%f[0-9]+]], [[D0]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[CF0:%f[0-9]+]], [[C0]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[DF1:%f[0-9]+]], [[D1]];
; CHECK-NOF16-DAG: cvt.f32.f16 [[CF1:%f[0-9]+]], [[C1]];
; CHECK-NOF16-DAG: setp.neu.f32    [[P0:%p[0-9]+]], [[CF0]], [[DF0]]
; CHECK-NOF16-DAG: setp.neu.f32    [[P1:%p[0-9]+]], [[CF1]], [[DF1]]
;
; CHECK-DAG: selp.f32        [[R0:%f[0-9]+]], [[A0]], [[B0]], [[P0]];
; CHECK-DAG: selp.f32        [[R1:%f[0-9]+]], [[A1]], [[B1]], [[P1]];
; CHECK-NEXT: st.param.v2.f32    [func_retval0+0], {[[R0]], [[R1]]};
; CHECK-NEXT: ret;
define <2 x float> @test_select_cc_f32_f16(<2 x float> %a, <2 x float> %b,
                                           <2 x half> %c, <2 x half> %d) #0 {
  %cc = fcmp une <2 x half> %c, %d
  %r = select <2 x i1> %cc, <2 x float> %a, <2 x float> %b
  ret <2 x float> %r
}

; CHECK-LABEL: test_select_cc_f16_f32(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_select_cc_f16_f32_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_select_cc_f16_f32_param_1];
; CHECK-DAG:  ld.param.v2.f32 {[[C0:%f[0-9]+]], [[C1:%f[0-9]+]]}, [test_select_cc_f16_f32_param_2];
; CHECK-DAG:  ld.param.v2.f32 {[[D0:%f[0-9]+]], [[D1:%f[0-9]+]]}, [test_select_cc_f16_f32_param_3];
; CHECK-DAG:  setp.neu.f32    [[P0:%p[0-9]+]], [[C0]], [[D0]]
; CHECK-DAG:  setp.neu.f32    [[P1:%p[0-9]+]], [[C1]], [[D1]]
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  mov.b32         {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-DAG:  selp.b16        [[R0:%rs[0-9]+]], [[A0]], [[B0]], [[P0]];
; CHECK-DAG:  selp.b16        [[R1:%rs[0-9]+]], [[A1]], [[B1]], [[P1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x half> @test_select_cc_f16_f32(<2 x half> %a, <2 x half> %b,
                                          <2 x float> %c, <2 x float> %d) #0 {
  %cc = fcmp une <2 x float> %c, %d
  %r = select <2 x i1> %cc, <2 x half> %a, <2 x half> %b
  ret <2 x half> %r
}

; CHECK-LABEL: test_fcmp_une(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fcmp_une_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fcmp_une_param_1];
; CHECK-F16:  setp.neu.f16x2  [[P0:%p[0-9]+]]|[[P1:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  setp.neu.f32   [[P0:%p[0-9]+]], [[FA0]], [[FB0]]
; CHECK-NOF16-DAG:  setp.neu.f32   [[P1:%p[0-9]+]], [[FA1]], [[FB1]]
; CHECK-DAG:  selp.u16        [[R0:%rs[0-9]+]], -1, 0, [[P0]];
; CHECK-NEXT: st.param.b8  [func_retval0+0], [[R0]];
; CHECK-DAG:  selp.u16        [[R1:%rs[0-9]+]], -1, 0, [[P1]];
; CHECK-NEXT: st.param.b8  [func_retval0+1], [[R1]];
; CHECK-NEXT: ret;
define <2 x i1> @test_fcmp_une(<2 x half> %a, <2 x half> %b) #0 {
  %r = fcmp une <2 x half> %a, %b
  ret <2 x i1> %r
}

; CHECK-LABEL: test_fcmp_ueq(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fcmp_ueq_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fcmp_ueq_param_1];
; CHECK-F16:  setp.equ.f16x2  [[P0:%p[0-9]+]]|[[P1:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  setp.equ.f32   [[P0:%p[0-9]+]], [[FA0]], [[FB0]]
; CHECK-NOF16-DAG:  setp.equ.f32   [[P1:%p[0-9]+]], [[FA1]], [[FB1]]
; CHECK-DAG:  selp.u16        [[R0:%rs[0-9]+]], -1, 0, [[P0]];
; CHECK-NEXT: st.param.b8     [func_retval0+0], [[R0]];
; CHECK-DAG:  selp.u16        [[R1:%rs[0-9]+]], -1, 0, [[P1]];
; CHECK-NEXT: st.param.b8     [func_retval0+1], [[R1]];
; CHECK-NEXT: ret;
define <2 x i1> @test_fcmp_ueq(<2 x half> %a, <2 x half> %b) #0 {
  %r = fcmp ueq <2 x half> %a, %b
  ret <2 x i1> %r
}

; CHECK-LABEL: test_fcmp_ugt(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fcmp_ugt_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fcmp_ugt_param_1];
; CHECK-F16:  setp.gtu.f16x2  [[P0:%p[0-9]+]]|[[P1:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  setp.gtu.f32   [[P0:%p[0-9]+]], [[FA0]], [[FB0]]
; CHECK-NOF16-DAG:  setp.gtu.f32   [[P1:%p[0-9]+]], [[FA1]], [[FB1]]
; CHECK-DAG:  selp.u16        [[R0:%rs[0-9]+]], -1, 0, [[P0]];
; CHECK-NEXT: st.param.b8     [func_retval0+0], [[R0]];
; CHECK-DAG:  selp.u16        [[R1:%rs[0-9]+]], -1, 0, [[P1]];
; CHECK-NEXT: st.param.b8     [func_retval0+1], [[R1]];
; CHECK-NEXT: ret;
define <2 x i1> @test_fcmp_ugt(<2 x half> %a, <2 x half> %b) #0 {
  %r = fcmp ugt <2 x half> %a, %b
  ret <2 x i1> %r
}

; CHECK-LABEL: test_fcmp_uge(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fcmp_uge_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fcmp_uge_param_1];
; CHECK-F16:  setp.geu.f16x2  [[P0:%p[0-9]+]]|[[P1:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  setp.geu.f32   [[P0:%p[0-9]+]], [[FA0]], [[FB0]]
; CHECK-NOF16-DAG:  setp.geu.f32   [[P1:%p[0-9]+]], [[FA1]], [[FB1]]
; CHECK-DAG:  selp.u16        [[R0:%rs[0-9]+]], -1, 0, [[P0]];
; CHECK-NEXT: st.param.b8     [func_retval0+0], [[R0]];
; CHECK-DAG:  selp.u16        [[R1:%rs[0-9]+]], -1, 0, [[P1]];
; CHECK-NEXT: st.param.b8     [func_retval0+1], [[R1]];
; CHECK-NEXT: ret;
define <2 x i1> @test_fcmp_uge(<2 x half> %a, <2 x half> %b) #0 {
  %r = fcmp uge <2 x half> %a, %b
  ret <2 x i1> %r
}

; CHECK-LABEL: test_fcmp_ult(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fcmp_ult_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fcmp_ult_param_1];
; CHECK-F16:  setp.ltu.f16x2  [[P0:%p[0-9]+]]|[[P1:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  setp.ltu.f32   [[P0:%p[0-9]+]], [[FA0]], [[FB0]]
; CHECK-NOF16-DAG:  setp.ltu.f32   [[P1:%p[0-9]+]], [[FA1]], [[FB1]]
; CHECK-DAG:  selp.u16        [[R0:%rs[0-9]+]], -1, 0, [[P0]];
; CHECK-NEXT: st.param.b8     [func_retval0+0], [[R0]];
; CHECK-DAG:  selp.u16        [[R1:%rs[0-9]+]], -1, 0, [[P1]];
; CHECK-NEXT: st.param.b8     [func_retval0+1], [[R1]];
; CHECK-NEXT: ret;
define <2 x i1> @test_fcmp_ult(<2 x half> %a, <2 x half> %b) #0 {
  %r = fcmp ult <2 x half> %a, %b
  ret <2 x i1> %r
}

; CHECK-LABEL: test_fcmp_ule(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fcmp_ule_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fcmp_ule_param_1];
; CHECK-F16:  setp.leu.f16x2  [[P0:%p[0-9]+]]|[[P1:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  setp.leu.f32   [[P0:%p[0-9]+]], [[FA0]], [[FB0]]
; CHECK-NOF16-DAG:  setp.leu.f32   [[P1:%p[0-9]+]], [[FA1]], [[FB1]]
; CHECK-DAG:  selp.u16        [[R0:%rs[0-9]+]], -1, 0, [[P0]];
; CHECK-NEXT: st.param.b8     [func_retval0+0], [[R0]];
; CHECK-DAG:  selp.u16        [[R1:%rs[0-9]+]], -1, 0, [[P1]];
; CHECK-NEXT: st.param.b8     [func_retval0+1], [[R1]];
; CHECK-NEXT: ret;
define <2 x i1> @test_fcmp_ule(<2 x half> %a, <2 x half> %b) #0 {
  %r = fcmp ule <2 x half> %a, %b
  ret <2 x i1> %r
}


; CHECK-LABEL: test_fcmp_uno(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fcmp_uno_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fcmp_uno_param_1];
; CHECK-F16:  setp.nan.f16x2  [[P0:%p[0-9]+]]|[[P1:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  setp.nan.f32   [[P0:%p[0-9]+]], [[FA0]], [[FB0]]
; CHECK-NOF16-DAG:  setp.nan.f32   [[P1:%p[0-9]+]], [[FA1]], [[FB1]]
; CHECK-DAG:  selp.u16        [[R0:%rs[0-9]+]], -1, 0, [[P0]];
; CHECK-NEXT: st.param.b8     [func_retval0+0], [[R0]];
; CHECK-DAG:  selp.u16        [[R1:%rs[0-9]+]], -1, 0, [[P1]];
; CHECK-NEXT: st.param.b8     [func_retval0+1], [[R1]];
; CHECK-NEXT: ret;
define <2 x i1> @test_fcmp_uno(<2 x half> %a, <2 x half> %b) #0 {
  %r = fcmp uno <2 x half> %a, %b
  ret <2 x i1> %r
}

; CHECK-LABEL: test_fcmp_one(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fcmp_one_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fcmp_one_param_1];
; CHECK-F16:  setp.ne.f16x2  [[P0:%p[0-9]+]]|[[P1:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  setp.ne.f32   [[P0:%p[0-9]+]], [[FA0]], [[FB0]]
; CHECK-NOF16-DAG:  setp.ne.f32   [[P1:%p[0-9]+]], [[FA1]], [[FB1]]
; CHECK-DAG:  selp.u16        [[R0:%rs[0-9]+]], -1, 0, [[P0]];
; CHECK-NEXT: st.param.b8     [func_retval0+0], [[R0]];
; CHECK-DAG:  selp.u16        [[R1:%rs[0-9]+]], -1, 0, [[P1]];
; CHECK-NEXT: st.param.b8     [func_retval0+1], [[R1]];
; CHECK-NEXT: ret;
define <2 x i1> @test_fcmp_one(<2 x half> %a, <2 x half> %b) #0 {
  %r = fcmp one <2 x half> %a, %b
  ret <2 x i1> %r
}

; CHECK-LABEL: test_fcmp_oeq(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fcmp_oeq_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fcmp_oeq_param_1];
; CHECK-F16:  setp.eq.f16x2  [[P0:%p[0-9]+]]|[[P1:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  setp.eq.f32   [[P0:%p[0-9]+]], [[FA0]], [[FB0]]
; CHECK-NOF16-DAG:  setp.eq.f32   [[P1:%p[0-9]+]], [[FA1]], [[FB1]]
; CHECK-DAG:  selp.u16        [[R0:%rs[0-9]+]], -1, 0, [[P0]];
; CHECK-NEXT: st.param.b8     [func_retval0+0], [[R0]];
; CHECK-DAG:  selp.u16        [[R1:%rs[0-9]+]], -1, 0, [[P1]];
; CHECK-NEXT: st.param.b8     [func_retval0+1], [[R1]];
; CHECK-NEXT: ret;
define <2 x i1> @test_fcmp_oeq(<2 x half> %a, <2 x half> %b) #0 {
  %r = fcmp oeq <2 x half> %a, %b
  ret <2 x i1> %r
}

; CHECK-LABEL: test_fcmp_ogt(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fcmp_ogt_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fcmp_ogt_param_1];
; CHECK-F16:  setp.gt.f16x2  [[P0:%p[0-9]+]]|[[P1:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  setp.gt.f32   [[P0:%p[0-9]+]], [[FA0]], [[FB0]]
; CHECK-NOF16-DAG:  setp.gt.f32   [[P1:%p[0-9]+]], [[FA1]], [[FB1]]
; CHECK-DAG:  selp.u16        [[R0:%rs[0-9]+]], -1, 0, [[P0]];
; CHECK-NEXT: st.param.b8     [func_retval0+0], [[R0]];
; CHECK-DAG:  selp.u16        [[R1:%rs[0-9]+]], -1, 0, [[P1]];
; CHECK-NEXT: st.param.b8     [func_retval0+1], [[R1]];
; CHECK-NEXT: ret;
define <2 x i1> @test_fcmp_ogt(<2 x half> %a, <2 x half> %b) #0 {
  %r = fcmp ogt <2 x half> %a, %b
  ret <2 x i1> %r
}

; CHECK-LABEL: test_fcmp_oge(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fcmp_oge_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fcmp_oge_param_1];
; CHECK-F16:  setp.ge.f16x2  [[P0:%p[0-9]+]]|[[P1:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  setp.ge.f32   [[P0:%p[0-9]+]], [[FA0]], [[FB0]]
; CHECK-NOF16-DAG:  setp.ge.f32   [[P1:%p[0-9]+]], [[FA1]], [[FB1]]
; CHECK-DAG:  selp.u16        [[R0:%rs[0-9]+]], -1, 0, [[P0]];
; CHECK-NEXT: st.param.b8     [func_retval0+0], [[R0]];
; CHECK-DAG:  selp.u16        [[R1:%rs[0-9]+]], -1, 0, [[P1]];
; CHECK-NEXT: st.param.b8     [func_retval0+1], [[R1]];
; CHECK-NEXT: ret;
define <2 x i1> @test_fcmp_oge(<2 x half> %a, <2 x half> %b) #0 {
  %r = fcmp oge <2 x half> %a, %b
  ret <2 x i1> %r
}

; CHECK-LABEL: test_fcmp_olt(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fcmp_olt_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fcmp_olt_param_1];
; CHECK-F16:  setp.lt.f16x2  [[P0:%p[0-9]+]]|[[P1:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  setp.lt.f32   [[P0:%p[0-9]+]], [[FA0]], [[FB0]]
; CHECK-NOF16-DAG:  setp.lt.f32   [[P1:%p[0-9]+]], [[FA1]], [[FB1]]
; CHECK-DAG:  selp.u16        [[R0:%rs[0-9]+]], -1, 0, [[P0]];
; CHECK-NEXT: st.param.b8     [func_retval0+0], [[R0]];
; CHECK-DAG:  selp.u16        [[R1:%rs[0-9]+]], -1, 0, [[P1]];
; CHECK-NEXT: st.param.b8     [func_retval0+1], [[R1]];
; CHECK-NEXT: ret;
define <2 x i1> @test_fcmp_olt(<2 x half> %a, <2 x half> %b) #0 {
  %r = fcmp olt <2 x half> %a, %b
  ret <2 x i1> %r
}

; XCHECK-LABEL: test_fcmp_ole(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fcmp_ole_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fcmp_ole_param_1];
; CHECK-F16:  setp.le.f16x2  [[P0:%p[0-9]+]]|[[P1:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  setp.le.f32   [[P0:%p[0-9]+]], [[FA0]], [[FB0]]
; CHECK-NOF16-DAG:  setp.le.f32   [[P1:%p[0-9]+]], [[FA1]], [[FB1]]
; CHECK-DAG:  selp.u16        [[R0:%rs[0-9]+]], -1, 0, [[P0]];
; CHECK-NEXT: st.param.b8     [func_retval0+0], [[R0]];
; CHECK-DAG:  selp.u16        [[R1:%rs[0-9]+]], -1, 0, [[P1]];
; CHECK-NEXT: st.param.b8     [func_retval0+1], [[R1]];
; CHECK-NEXT: ret;
define <2 x i1> @test_fcmp_ole(<2 x half> %a, <2 x half> %b) #0 {
  %r = fcmp ole <2 x half> %a, %b
  ret <2 x i1> %r
}

; CHECK-LABEL: test_fcmp_ord(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fcmp_ord_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fcmp_ord_param_1];
; CHECK-F16:  setp.num.f16x2  [[P0:%p[0-9]+]]|[[P1:%p[0-9]+]], [[A]], [[B]]
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  setp.num.f32   [[P0:%p[0-9]+]], [[FA0]], [[FB0]]
; CHECK-NOF16-DAG:  setp.num.f32   [[P1:%p[0-9]+]], [[FA1]], [[FB1]]
; CHECK-DAG:  selp.u16        [[R0:%rs[0-9]+]], -1, 0, [[P0]];
; CHECK-NEXT: st.param.b8     [func_retval0+0], [[R0]];
; CHECK-DAG:  selp.u16        [[R1:%rs[0-9]+]], -1, 0, [[P1]];
; CHECK-NEXT: st.param.b8     [func_retval0+1], [[R1]];
; CHECK-NEXT: ret;
define <2 x i1> @test_fcmp_ord(<2 x half> %a, <2 x half> %b) #0 {
  %r = fcmp ord <2 x half> %a, %b
  ret <2 x i1> %r
}

; CHECK-LABEL: test_fptosi_i32(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_fptosi_i32_param_0];
; CHECK:      mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  cvt.rzi.s32.f16 [[R0:%r[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.rzi.s32.f16 [[R1:%r[0-9]+]], [[A1]];
; CHECK:      st.param.v2.b32 [func_retval0+0], {[[R0]], [[R1]]}
; CHECK:      ret;
define <2 x i32> @test_fptosi_i32(<2 x half> %a) #0 {
  %r = fptosi <2 x half> %a to <2 x i32>
  ret <2 x i32> %r
}

; CHECK-LABEL: test_fptosi_i64(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_fptosi_i64_param_0];
; CHECK:      mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  cvt.rzi.s64.f16 [[R0:%rd[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.rzi.s64.f16 [[R1:%rd[0-9]+]], [[A1]];
; CHECK:      st.param.v2.b64 [func_retval0+0], {[[R0]], [[R1]]}
; CHECK:      ret;
define <2 x i64> @test_fptosi_i64(<2 x half> %a) #0 {
  %r = fptosi <2 x half> %a to <2 x i64>
  ret <2 x i64> %r
}

; CHECK-LABEL: test_fptoui_2xi32(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_fptoui_2xi32_param_0];
; CHECK:      mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  cvt.rzi.u32.f16 [[R0:%r[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.rzi.u32.f16 [[R1:%r[0-9]+]], [[A1]];
; CHECK:      st.param.v2.b32 [func_retval0+0], {[[R0]], [[R1]]}
; CHECK:      ret;
define <2 x i32> @test_fptoui_2xi32(<2 x half> %a) #0 {
  %r = fptoui <2 x half> %a to <2 x i32>
  ret <2 x i32> %r
}

; CHECK-LABEL: test_fptoui_2xi64(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_fptoui_2xi64_param_0];
; CHECK:      mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  cvt.rzi.u64.f16 [[R0:%rd[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.rzi.u64.f16 [[R1:%rd[0-9]+]], [[A1]];
; CHECK:      st.param.v2.b64 [func_retval0+0], {[[R0]], [[R1]]}
; CHECK:      ret;
define <2 x i64> @test_fptoui_2xi64(<2 x half> %a) #0 {
  %r = fptoui <2 x half> %a to <2 x i64>
  ret <2 x i64> %r
}

; CHECK-LABEL: test_uitofp_2xi32(
; CHECK:      ld.param.v2.u32 {[[A0:%r[0-9]+]], [[A1:%r[0-9]+]]}, [test_uitofp_2xi32_param_0];
; CHECK-DAG:  cvt.rn.f16.u32  [[R0:%rs[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.rn.f16.u32  [[R1:%rs[0-9]+]], [[A1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_uitofp_2xi32(<2 x i32> %a) #0 {
  %r = uitofp <2 x i32> %a to <2 x half>
  ret <2 x half> %r
}

; CHECK-LABEL: test_uitofp_2xi64(
; CHECK:      ld.param.v2.u64 {[[A0:%rd[0-9]+]], [[A1:%rd[0-9]+]]}, [test_uitofp_2xi64_param_0];
; CHECK-DAG:  cvt.rn.f16.u64  [[R0:%rs[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.rn.f16.u64  [[R1:%rs[0-9]+]], [[A1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_uitofp_2xi64(<2 x i64> %a) #0 {
  %r = uitofp <2 x i64> %a to <2 x half>
  ret <2 x half> %r
}

; CHECK-LABEL: test_sitofp_2xi32(
; CHECK:      ld.param.v2.u32 {[[A0:%r[0-9]+]], [[A1:%r[0-9]+]]}, [test_sitofp_2xi32_param_0];
; CHECK-DAG:  cvt.rn.f16.s32  [[R0:%rs[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.rn.f16.s32  [[R1:%rs[0-9]+]], [[A1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_sitofp_2xi32(<2 x i32> %a) #0 {
  %r = sitofp <2 x i32> %a to <2 x half>
  ret <2 x half> %r
}

; CHECK-LABEL: test_sitofp_2xi64(
; CHECK:      ld.param.v2.u64 {[[A0:%rd[0-9]+]], [[A1:%rd[0-9]+]]}, [test_sitofp_2xi64_param_0];
; CHECK-DAG:  cvt.rn.f16.s64  [[R0:%rs[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.rn.f16.s64  [[R1:%rs[0-9]+]], [[A1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_sitofp_2xi64(<2 x i64> %a) #0 {
  %r = sitofp <2 x i64> %a to <2 x half>
  ret <2 x half> %r
}

; CHECK-LABEL: test_uitofp_2xi32_fadd(
; CHECK-DAG:  ld.param.v2.u32 {[[A0:%r[0-9]+]], [[A1:%r[0-9]+]]}, [test_uitofp_2xi32_fadd_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_uitofp_2xi32_fadd_param_1];
; CHECK-DAG:  cvt.rn.f16.u32  [[C0:%rs[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.rn.f16.u32  [[C1:%rs[0-9]+]], [[A1]];

; CHECK-F16-DAG:  mov.b32         [[C:%r[0-9]+]], {[[C0]], [[C1]]}
; CHECK-F16-DAG:  add.rn.f16x2    [[R:%r[0-9]+]], [[B]], [[C]];
;
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FC0:%f[0-9]+]], [[C0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FC1:%f[0-9]+]], [[C1]]
; CHECK-NOF16-DAG:  add.rn.f32     [[FR0:%f[0-9]+]], [[FB0]], [[FC0]];
; CHECK-NOF16-DAG:  add.rn.f32     [[FR1:%f[0-9]+]], [[FB1]], [[FC1]];
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R0:%rs[0-9]+]], [[FR0]]
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R1:%rs[0-9]+]], [[FR1]]
; CHECK-NOF16:      mov.b32        [[R:%r[0-9]+]], {[[R0]], [[R1]]}
;
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_uitofp_2xi32_fadd(<2 x i32> %a, <2 x half> %b) #0 {
  %c = uitofp <2 x i32> %a to <2 x half>
  %r = fadd <2 x half> %b, %c
  ret <2 x half> %r
}

; CHECK-LABEL: test_sitofp_2xi32_fadd(
; CHECK-DAG:  ld.param.v2.u32 {[[A0:%r[0-9]+]], [[A1:%r[0-9]+]]}, [test_sitofp_2xi32_fadd_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_sitofp_2xi32_fadd_param_1];
; CHECK-DAG:  cvt.rn.f16.s32  [[C0:%rs[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.rn.f16.s32  [[C1:%rs[0-9]+]], [[A1]];
;
; CHECK-F16-DAG:  mov.b32         [[C:%r[0-9]+]], {[[C0]], [[C1]]}
; CHECK-F16-DAG:  add.rn.f16x2    [[R:%r[0-9]+]], [[B]], [[C]];
;
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FC0:%f[0-9]+]], [[C0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FC1:%f[0-9]+]], [[C1]]
; CHECK-NOF16-DAG:  add.rn.f32     [[FR0:%f[0-9]+]], [[FB0]], [[FC0]];
; CHECK-NOF16-DAG:  add.rn.f32     [[FR1:%f[0-9]+]], [[FB1]], [[FC1]];
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R0:%rs[0-9]+]], [[FR0]]
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R1:%rs[0-9]+]], [[FR1]]
; CHECK-NOF16:      mov.b32        [[R:%r[0-9]+]], {[[R0]], [[R1]]}
;
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_sitofp_2xi32_fadd(<2 x i32> %a, <2 x half> %b) #0 {
  %c = sitofp <2 x i32> %a to <2 x half>
  %r = fadd <2 x half> %b, %c
  ret <2 x half> %r
}

; CHECK-LABEL: test_fptrunc_2xfloat(
; CHECK:      ld.param.v2.f32 {[[A0:%f[0-9]+]], [[A1:%f[0-9]+]]}, [test_fptrunc_2xfloat_param_0];
; CHECK-DAG:  cvt.rn.f16.f32  [[R0:%rs[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.rn.f16.f32  [[R1:%rs[0-9]+]], [[A1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_fptrunc_2xfloat(<2 x float> %a) #0 {
  %r = fptrunc <2 x float> %a to <2 x half>
  ret <2 x half> %r
}

; CHECK-LABEL: test_fptrunc_2xdouble(
; CHECK:      ld.param.v2.f64 {[[A0:%fd[0-9]+]], [[A1:%fd[0-9]+]]}, [test_fptrunc_2xdouble_param_0];
; CHECK-DAG:  cvt.rn.f16.f64  [[R0:%rs[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.rn.f16.f64  [[R1:%rs[0-9]+]], [[A1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_fptrunc_2xdouble(<2 x double> %a) #0 {
  %r = fptrunc <2 x double> %a to <2 x half>
  ret <2 x half> %r
}

; CHECK-LABEL: test_fpext_2xfloat(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_fpext_2xfloat_param_0];
; CHECK:      mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  cvt.f32.f16     [[R0:%f[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.f32.f16     [[R1:%f[0-9]+]], [[A1]];
; CHECK-NEXT: st.param.v2.f32 [func_retval0+0], {[[R0]], [[R1]]};
; CHECK:      ret;
define <2 x float> @test_fpext_2xfloat(<2 x half> %a) #0 {
  %r = fpext <2 x half> %a to <2 x float>
  ret <2 x float> %r
}

; CHECK-LABEL: test_fpext_2xdouble(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_fpext_2xdouble_param_0];
; CHECK:      mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  cvt.f64.f16     [[R0:%fd[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.f64.f16     [[R1:%fd[0-9]+]], [[A1]];
; CHECK-NEXT: st.param.v2.f64 [func_retval0+0], {[[R0]], [[R1]]};
; CHECK:      ret;
define <2 x double> @test_fpext_2xdouble(<2 x half> %a) #0 {
  %r = fpext <2 x half> %a to <2 x double>
  ret <2 x double> %r
}


; CHECK-LABEL: test_bitcast_2xhalf_to_2xi16(
; CHECK:      ld.param.u32    [[A:%r[0-9]+]], [test_bitcast_2xhalf_to_2xi16_param_0];
; CHECK:      st.param.b32 [func_retval0+0], [[A]]
; CHECK:      ret;
define <2 x i16> @test_bitcast_2xhalf_to_2xi16(<2 x half> %a) #0 {
  %r = bitcast <2 x half> %a to <2 x i16>
  ret <2 x i16> %r
}

; CHECK-LABEL: test_bitcast_2xi16_to_2xhalf(
; CHECK:      ld.param.u32         [[R:%r[0-9]+]], [test_bitcast_2xi16_to_2xhalf_param_0];
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_bitcast_2xi16_to_2xhalf(<2 x i16> %a) #0 {
  %r = bitcast <2 x i16> %a to <2 x half>
  ret <2 x half> %r
}

; CHECK-LABEL: test_bitcast_float_to_2xhalf(
; CHECK: ld.param.f32 	[[AF1:%f[0-9]+]], [test_bitcast_float_to_2xhalf_param_0];
; CHECK: mov.b32 	[[R:%r[0-9]+]], [[AF1]];
; CHECK: st.param.b32 	[func_retval0+0], [[R]];
; CHECK: ret;
define <2 x half> @test_bitcast_float_to_2xhalf(float %a) #0 {
  %r = bitcast float %a to <2 x half>
  ret <2 x half> %r
}

; CHECK-LABEL: test_bitcast_2xhalf_to_float(
; CHECK: ld.param.u32 	[[R:%r[0-9]+]], [test_bitcast_2xhalf_to_float_param_0];
; CHECK: mov.b32 	[[AF1:%f[0-9]+]], [[R]];
; CHECK: st.param.f32 	[func_retval0+0], [[AF1]];
; CHECK: ret;
define float @test_bitcast_2xhalf_to_float(<2 x half> %a) #0 {
  %r = bitcast <2 x half> %a to float
  ret float %r
}

declare <2 x half> @llvm.sqrt.f16(<2 x half> %a) #0
declare <2 x half> @llvm.powi.f16.i32(<2 x half> %a, <2 x i32> %b) #0
declare <2 x half> @llvm.sin.f16(<2 x half> %a) #0
declare <2 x half> @llvm.cos.f16(<2 x half> %a) #0
declare <2 x half> @llvm.pow.f16(<2 x half> %a, <2 x half> %b) #0
declare <2 x half> @llvm.exp.f16(<2 x half> %a) #0
declare <2 x half> @llvm.exp2.f16(<2 x half> %a) #0
declare <2 x half> @llvm.log.f16(<2 x half> %a) #0
declare <2 x half> @llvm.log10.f16(<2 x half> %a) #0
declare <2 x half> @llvm.log2.f16(<2 x half> %a) #0
declare <2 x half> @llvm.fma.f16(<2 x half> %a, <2 x half> %b, <2 x half> %c) #0
declare <2 x half> @llvm.fabs.f16(<2 x half> %a) #0
declare <2 x half> @llvm.minnum.f16(<2 x half> %a, <2 x half> %b) #0
declare <2 x half> @llvm.maxnum.f16(<2 x half> %a, <2 x half> %b) #0
declare <2 x half> @llvm.copysign.f16(<2 x half> %a, <2 x half> %b) #0
declare <2 x half> @llvm.floor.f16(<2 x half> %a) #0
declare <2 x half> @llvm.ceil.f16(<2 x half> %a) #0
declare <2 x half> @llvm.trunc.f16(<2 x half> %a) #0
declare <2 x half> @llvm.rint.f16(<2 x half> %a) #0
declare <2 x half> @llvm.nearbyint.f16(<2 x half> %a) #0
declare <2 x half> @llvm.round.f16(<2 x half> %a) #0
declare <2 x half> @llvm.roundeven.f16(<2 x half> %a) #0
declare <2 x half> @llvm.fmuladd.f16(<2 x half> %a, <2 x half> %b, <2 x half> %c) #0

; CHECK-LABEL: test_sqrt(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_sqrt_param_0];
; CHECK:      mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  cvt.f32.f16     [[AF0:%f[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.f32.f16     [[AF1:%f[0-9]+]], [[A1]];
; CHECK-DAG:  sqrt.rn.f32     [[RF0:%f[0-9]+]], [[AF0]];
; CHECK-DAG:  sqrt.rn.f32     [[RF1:%f[0-9]+]], [[AF1]];
; CHECK-DAG:  cvt.rn.f16.f32  [[R0:%rs[0-9]+]], [[RF0]];
; CHECK-DAG:  cvt.rn.f16.f32  [[R1:%rs[0-9]+]], [[RF1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_sqrt(<2 x half> %a) #0 {
  %r = call <2 x half> @llvm.sqrt.f16(<2 x half> %a)
  ret <2 x half> %r
}

;;; Can't do this yet: requires libcall.
; XCHECK-LABEL: test_powi(
;define <2 x half> @test_powi(<2 x half> %a, <2 x i32> %b) #0 {
;  %r = call <2 x half> @llvm.powi.f16.i32(<2 x half> %a, <2 x i32> %b)
;  ret <2 x half> %r
;}

; CHECK-LABEL: test_sin(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_sin_param_0];
; CHECK:      mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  cvt.f32.f16     [[AF0:%f[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.f32.f16     [[AF1:%f[0-9]+]], [[A1]];
; CHECK-DAG:  sin.approx.f32  [[RF0:%f[0-9]+]], [[AF0]];
; CHECK-DAG:  sin.approx.f32  [[RF1:%f[0-9]+]], [[AF1]];
; CHECK-DAG:  cvt.rn.f16.f32  [[R0:%rs[0-9]+]], [[RF0]];
; CHECK-DAG:  cvt.rn.f16.f32  [[R1:%rs[0-9]+]], [[RF1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_sin(<2 x half> %a) #0 #1 {
  %r = call <2 x half> @llvm.sin.f16(<2 x half> %a)
  ret <2 x half> %r
}

; CHECK-LABEL: test_cos(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_cos_param_0];
; CHECK:      mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  cvt.f32.f16     [[AF0:%f[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.f32.f16     [[AF1:%f[0-9]+]], [[A1]];
; CHECK-DAG:  cos.approx.f32  [[RF0:%f[0-9]+]], [[AF0]];
; CHECK-DAG:  cos.approx.f32  [[RF1:%f[0-9]+]], [[AF1]];
; CHECK-DAG:  cvt.rn.f16.f32  [[R0:%rs[0-9]+]], [[RF0]];
; CHECK-DAG:  cvt.rn.f16.f32  [[R1:%rs[0-9]+]], [[RF1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_cos(<2 x half> %a) #0 #1 {
  %r = call <2 x half> @llvm.cos.f16(<2 x half> %a)
  ret <2 x half> %r
}

;;; Can't do this yet: requires libcall.
; XCHECK-LABEL: test_pow(
;define <2 x half> @test_pow(<2 x half> %a, <2 x half> %b) #0 {
;  %r = call <2 x half> @llvm.pow.f16(<2 x half> %a, <2 x half> %b)
;  ret <2 x half> %r
;}

;;; Can't do this yet: requires libcall.
; XCHECK-LABEL: test_exp(
;define <2 x half> @test_exp(<2 x half> %a) #0 {
;  %r = call <2 x half> @llvm.exp.f16(<2 x half> %a)
;  ret <2 x half> %r
;}

;;; Can't do this yet: requires libcall.
; XCHECK-LABEL: test_exp2(
;define <2 x half> @test_exp2(<2 x half> %a) #0 {
;  %r = call <2 x half> @llvm.exp2.f16(<2 x half> %a)
;  ret <2 x half> %r
;}

;;; Can't do this yet: requires libcall.
; XCHECK-LABEL: test_log(
;define <2 x half> @test_log(<2 x half> %a) #0 {
;  %r = call <2 x half> @llvm.log.f16(<2 x half> %a)
;  ret <2 x half> %r
;}

;;; Can't do this yet: requires libcall.
; XCHECK-LABEL: test_log10(
;define <2 x half> @test_log10(<2 x half> %a) #0 {
;  %r = call <2 x half> @llvm.log10.f16(<2 x half> %a)
;  ret <2 x half> %r
;}

;;; Can't do this yet: requires libcall.
; XCHECK-LABEL: test_log2(
;define <2 x half> @test_log2(<2 x half> %a) #0 {
;  %r = call <2 x half> @llvm.log2.f16(<2 x half> %a)
;  ret <2 x half> %r
;}

; CHECK-LABEL: test_fma(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fma_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fma_param_1];
; CHECK-DAG:  ld.param.b32    [[C:%r[0-9]+]], [test_fma_param_2];
;
; CHECK-F16:        fma.rn.f16x2   [[R:%r[0-9]+]], [[A]], [[B]], [[C]];
;
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  mov.b32        {[[C0:%rs[0-9]+]], [[C1:%rs[0-9]+]]}, [[C]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FC0:%f[0-9]+]], [[C0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FC0:%f[0-9]+]], [[C0]]
; CHECK-NOF16-DAG:  fma.rn.f32     [[FR0:%f[0-9]+]], [[FA0]], [[FB0]], [[FC0]];
; CHECK-NOF16-DAG:  fma.rn.f32     [[FR1:%f[0-9]+]], [[FA1]], [[FB1]], [[FC1]];
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R0:%rs[0-9]+]], [[FR0]]
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R1:%rs[0-9]+]], [[FR1]]
; CHECK-NOF16:      mov.b32        [[R:%r[0-9]+]], {[[R0]], [[R1]]}

; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret
define <2 x half> @test_fma(<2 x half> %a, <2 x half> %b, <2 x half> %c) #0 {
  %r = call <2 x half> @llvm.fma.f16(<2 x half> %a, <2 x half> %b, <2 x half> %c)
  ret <2 x half> %r
}

; CHECK-LABEL: test_fabs(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_fabs_param_0];
; CHECK:      mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  cvt.f32.f16     [[AF0:%f[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.f32.f16     [[AF1:%f[0-9]+]], [[A1]];
; CHECK-DAG:  abs.f32         [[RF0:%f[0-9]+]], [[AF0]];
; CHECK-DAG:  abs.f32         [[RF1:%f[0-9]+]], [[AF1]];
; CHECK-DAG:  cvt.rn.f16.f32  [[R0:%rs[0-9]+]], [[RF0]];
; CHECK-DAG:  cvt.rn.f16.f32  [[R1:%rs[0-9]+]], [[RF1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_fabs(<2 x half> %a) #0 {
  %r = call <2 x half> @llvm.fabs.f16(<2 x half> %a)
  ret <2 x half> %r
}

; CHECK-LABEL: test_minnum(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_minnum_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_minnum_param_1];
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  mov.b32         {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-DAG:  cvt.f32.f16     [[AF0:%f[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.f32.f16     [[AF1:%f[0-9]+]], [[A1]];
; CHECK-DAG:  cvt.f32.f16     [[BF0:%f[0-9]+]], [[B0]];
; CHECK-DAG:  cvt.f32.f16     [[BF1:%f[0-9]+]], [[B1]];
; CHECK-DAG:  min.f32         [[RF0:%f[0-9]+]], [[AF0]], [[BF0]];
; CHECK-DAG:  min.f32         [[RF1:%f[0-9]+]], [[AF1]], [[BF1]];
; CHECK-DAG:  cvt.rn.f16.f32  [[R0:%rs[0-9]+]], [[RF0]];
; CHECK-DAG:  cvt.rn.f16.f32  [[R1:%rs[0-9]+]], [[RF1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_minnum(<2 x half> %a, <2 x half> %b) #0 {
  %r = call <2 x half> @llvm.minnum.f16(<2 x half> %a, <2 x half> %b)
  ret <2 x half> %r
}

; CHECK-LABEL: test_maxnum(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_maxnum_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_maxnum_param_1];
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  mov.b32         {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-DAG:  cvt.f32.f16     [[AF0:%f[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.f32.f16     [[AF1:%f[0-9]+]], [[A1]];
; CHECK-DAG:  cvt.f32.f16     [[BF0:%f[0-9]+]], [[B0]];
; CHECK-DAG:  cvt.f32.f16     [[BF1:%f[0-9]+]], [[B1]];
; CHECK-DAG:  max.f32         [[RF0:%f[0-9]+]], [[AF0]], [[BF0]];
; CHECK-DAG:  max.f32         [[RF1:%f[0-9]+]], [[AF1]], [[BF1]];
; CHECK-DAG:  cvt.rn.f16.f32  [[R0:%rs[0-9]+]], [[RF0]];
; CHECK-DAG:  cvt.rn.f16.f32  [[R1:%rs[0-9]+]], [[RF1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_maxnum(<2 x half> %a, <2 x half> %b) #0 {
  %r = call <2 x half> @llvm.maxnum.f16(<2 x half> %a, <2 x half> %b)
  ret <2 x half> %r
}

; CHECK-LABEL: test_copysign(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_copysign_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_copysign_param_1];
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  mov.b32         {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-DAG:  and.b16         [[AX0:%rs[0-9]+]], [[A0]], 32767;
; CHECK-DAG:  and.b16         [[AX1:%rs[0-9]+]], [[A1]], 32767;
; CHECK-DAG:  and.b16         [[BX0:%rs[0-9]+]], [[B0]], -32768;
; CHECK-DAG:  and.b16         [[BX1:%rs[0-9]+]], [[B1]], -32768;
; CHECK-DAG:  or.b16          [[R0:%rs[0-9]+]], [[AX0]], [[BX0]];
; CHECK-DAG:  or.b16          [[R1:%rs[0-9]+]], [[AX1]], [[BX1]];
; CHECK-DAG:  mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_copysign(<2 x half> %a, <2 x half> %b) #0 {
  %r = call <2 x half> @llvm.copysign.f16(<2 x half> %a, <2 x half> %b)
  ret <2 x half> %r
}

; CHECK-LABEL: test_copysign_f32(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_copysign_f32_param_0];
; CHECK-DAG:  ld.param.v2.f32 {[[B0:%f[0-9]+]], [[B1:%f[0-9]+]]}, [test_copysign_f32_param_1];
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  mov.b32         [[BI0:%r[0-9]+]], [[B0]];
; CHECK-DAG:  mov.b32         [[BI1:%r[0-9]+]], [[B1]];
; CHECK-DAG:  and.b16         [[AI0:%rs[0-9]+]], [[A0]], 32767;
; CHECK-DAG:  and.b16         [[AI1:%rs[0-9]+]], [[A1]], 32767;
; CHECK-DAG:  and.b32         [[BX0:%r[0-9]+]], [[BI0]], -2147483648;
; CHECK-DAG:  and.b32         [[BX1:%r[0-9]+]], [[BI1]], -2147483648;
; CHECK-DAG:  mov.b32         {tmp, [[BZ0:%rs[0-9]+]]}, [[BX0]]; }
; CHECK-DAG:  mov.b32         {tmp, [[BZ1:%rs[0-9]+]]}, [[BX1]]; }
; CHECK-DAG:  or.b16          [[R0:%rs[0-9]+]], [[AI0]], [[BZ0]];
; CHECK-DAG:  or.b16          [[R1:%rs[0-9]+]], [[AI1]], [[BZ1]];
; CHECK-DAG:  mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_copysign_f32(<2 x half> %a, <2 x float> %b) #0 {
  %tb = fptrunc <2 x float> %b to <2 x half>
  %r = call <2 x half> @llvm.copysign.f16(<2 x half> %a, <2 x half> %tb)
  ret <2 x half> %r
}

; CHECK-LABEL: test_copysign_f64(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_copysign_f64_param_0];
; CHECK-DAG:  ld.param.v2.f64 {[[B0:%fd[0-9]+]], [[B1:%fd[0-9]+]]}, [test_copysign_f64_param_1];
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  mov.b64         [[BI0:%rd[0-9]+]], [[B0]];
; CHECK-DAG:  mov.b64         [[BI1:%rd[0-9]+]], [[B1]];
; CHECK-DAG:  and.b16         [[AI0:%rs[0-9]+]], [[A0]], 32767;
; CHECK-DAG:  and.b16         [[AI1:%rs[0-9]+]], [[A1]], 32767;
; CHECK-DAG:  and.b64         [[BX0:%rd[0-9]+]], [[BI0]], -9223372036854775808;
; CHECK-DAG:  and.b64         [[BX1:%rd[0-9]+]], [[BI1]], -9223372036854775808;
; CHECK-DAG:  shr.u64         [[BY0:%rd[0-9]+]], [[BX0]], 48;
; CHECK-DAG:  shr.u64         [[BY1:%rd[0-9]+]], [[BX1]], 48;
; CHECK-DAG:  cvt.u16.u64     [[BZ0:%rs[0-9]+]], [[BY0]];
; CHECK-DAG:  cvt.u16.u64     [[BZ1:%rs[0-9]+]], [[BY1]];
; CHECK-DAG:  or.b16          [[R0:%rs[0-9]+]], [[AI0]], [[BZ0]];
; CHECK-DAG:  or.b16          [[R1:%rs[0-9]+]], [[AI1]], [[BZ1]];
; CHECK-DAG:  mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_copysign_f64(<2 x half> %a, <2 x double> %b) #0 {
  %tb = fptrunc <2 x double> %b to <2 x half>
  %r = call <2 x half> @llvm.copysign.f16(<2 x half> %a, <2 x half> %tb)
  ret <2 x half> %r
}

; CHECK-LABEL: test_copysign_extended(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_copysign_extended_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_copysign_extended_param_1];
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  mov.b32         {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-DAG:  and.b16         [[AX0:%rs[0-9]+]], [[A0]], 32767;
; CHECK-DAG:  and.b16         [[AX1:%rs[0-9]+]], [[A1]], 32767;
; CHECK-DAG:  and.b16         [[BX0:%rs[0-9]+]], [[B0]], -32768;
; CHECK-DAG:  and.b16         [[BX1:%rs[0-9]+]], [[B1]], -32768;
; CHECK-DAG:  or.b16          [[R0:%rs[0-9]+]], [[AX0]], [[BX0]];
; CHECK-DAG:  or.b16          [[R1:%rs[0-9]+]], [[AX1]], [[BX1]];
; CHECK-DAG:  mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      mov.b32         {[[RX0:%rs[0-9]+]], [[RX1:%rs[0-9]+]]}, [[R]]
; CHECK-DAG:  cvt.f32.f16     [[XR0:%f[0-9]+]], [[RX0]];
; CHECK-DAG:  cvt.f32.f16     [[XR1:%f[0-9]+]], [[RX1]];
; CHECK:      st.param.v2.f32 [func_retval0+0], {[[XR0]], [[XR1]]};
; CHECK:      ret;
define <2 x float> @test_copysign_extended(<2 x half> %a, <2 x half> %b) #0 {
  %r = call <2 x half> @llvm.copysign.f16(<2 x half> %a, <2 x half> %b)
  %xr = fpext <2 x half> %r to <2 x float>
  ret <2 x float> %xr
}

; CHECK-LABEL: test_floor(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_floor_param_0];
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]];
; CHECK-DAG:  cvt.rmi.f16.f16 [[R1:%rs[0-9]+]], [[A1]];
; CHECK-DAG:  cvt.rmi.f16.f16 [[R0:%rs[0-9]+]], [[A0]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_floor(<2 x half> %a) #0 {
  %r = call <2 x half> @llvm.floor.f16(<2 x half> %a)
  ret <2 x half> %r
}

; CHECK-LABEL: test_ceil(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_ceil_param_0];
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]];
; CHECK-DAG:  cvt.rpi.f16.f16 [[R1:%rs[0-9]+]], [[A1]];
; CHECK-DAG:  cvt.rpi.f16.f16 [[R0:%rs[0-9]+]], [[A0]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_ceil(<2 x half> %a) #0 {
  %r = call <2 x half> @llvm.ceil.f16(<2 x half> %a)
  ret <2 x half> %r
}

; CHECK-LABEL: test_trunc(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_trunc_param_0];
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]];
; CHECK-DAG:  cvt.rzi.f16.f16 [[R1:%rs[0-9]+]], [[A1]];
; CHECK-DAG:  cvt.rzi.f16.f16 [[R0:%rs[0-9]+]], [[A0]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_trunc(<2 x half> %a) #0 {
  %r = call <2 x half> @llvm.trunc.f16(<2 x half> %a)
  ret <2 x half> %r
}

; CHECK-LABEL: test_rint(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_rint_param_0];
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]];
; CHECK-DAG:  cvt.rni.f16.f16 [[R1:%rs[0-9]+]], [[A1]];
; CHECK-DAG:  cvt.rni.f16.f16 [[R0:%rs[0-9]+]], [[A0]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_rint(<2 x half> %a) #0 {
  %r = call <2 x half> @llvm.rint.f16(<2 x half> %a)
  ret <2 x half> %r
}

; CHECK-LABEL: test_nearbyint(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_nearbyint_param_0];
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]];
; CHECK-DAG:  cvt.rni.f16.f16 [[R1:%rs[0-9]+]], [[A1]];
; CHECK-DAG:  cvt.rni.f16.f16 [[R0:%rs[0-9]+]], [[A0]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_nearbyint(<2 x half> %a) #0 {
  %r = call <2 x half> @llvm.nearbyint.f16(<2 x half> %a)
  ret <2 x half> %r
}

; CHECK-LABEL: test_roundeven(
; CHECK:      ld.param.b32    [[A:%r[0-9]+]], [test_roundeven_param_0];
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]];
; CHECK-DAG:  cvt.rni.f16.f16 [[R1:%rs[0-9]+]], [[A1]];
; CHECK-DAG:  cvt.rni.f16.f16 [[R0:%rs[0-9]+]], [[A0]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_roundeven(<2 x half> %a) #0 {
  %r = call <2 x half> @llvm.roundeven.f16(<2 x half> %a)
  ret <2 x half> %r
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
define <2 x half> @test_round(<2 x half> %a) #0 {
  %r = call <2 x half> @llvm.round.f16(<2 x half> %a)
  ret <2 x half> %r
}

; CHECK-LABEL: test_fmuladd(
; CHECK-DAG:  ld.param.b32    [[A:%r[0-9]+]], [test_fmuladd_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%r[0-9]+]], [test_fmuladd_param_1];
; CHECK-DAG:  ld.param.b32    [[C:%r[0-9]+]], [test_fmuladd_param_2];
;
; CHECK-F16:        fma.rn.f16x2   [[R:%r[0-9]+]], [[A]], [[B]], [[C]];
;
; CHECK-NOF16-DAG:  mov.b32        {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-NOF16-DAG:  mov.b32        {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-NOF16-DAG:  mov.b32        {[[C0:%rs[0-9]+]], [[C1:%rs[0-9]+]]}, [[C]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA0:%f[0-9]+]], [[A0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB0:%f[0-9]+]], [[B0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FC0:%f[0-9]+]], [[C0]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FA1:%f[0-9]+]], [[A1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FB1:%f[0-9]+]], [[B1]]
; CHECK-NOF16-DAG:  cvt.f32.f16    [[FC0:%f[0-9]+]], [[C0]]
; CHECK-NOF16-DAG:  fma.rn.f32     [[FR0:%f[0-9]+]], [[FA0]], [[FB0]], [[FC0]];
; CHECK-NOF16-DAG:  fma.rn.f32     [[FR1:%f[0-9]+]], [[FA1]], [[FB1]], [[FC1]];
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R0:%rs[0-9]+]], [[FR0]]
; CHECK-NOF16-DAG:  cvt.rn.f16.f32 [[R1:%rs[0-9]+]], [[FR1]]
; CHECK-NOF16:      mov.b32        [[R:%r[0-9]+]], {[[R0]], [[R1]]}
;
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x half> @test_fmuladd(<2 x half> %a, <2 x half> %b, <2 x half> %c) #0 {
  %r = call <2 x half> @llvm.fmuladd.f16(<2 x half> %a, <2 x half> %b, <2 x half> %c)
  ret <2 x half> %r
}

; CHECK-LABEL: test_shufflevector(
; CHECK: mov.b32 {%rs1, %rs2}, %r1;
; CHECK: mov.b32 %r2, {%rs2, %rs1};
define <2 x half> @test_shufflevector(<2 x half> %a) #0 {
  %s = shufflevector <2 x half> %a, <2 x half> undef, <2 x i32> <i32 1, i32 0>
  ret <2 x half> %s
}

; CHECK-LABEL: test_insertelement(
; CHECK: mov.b32 {%rs2, tmp}, %r1;
; CHECK: mov.b32 %r2, {%rs2, %rs1};
define <2 x half> @test_insertelement(<2 x half> %a, half %x) #0 {
  %i = insertelement <2 x half> %a, half %x, i64 1
  ret <2 x half> %i
}

attributes #0 = { nounwind }
attributes #1 = { "unsafe-fp-math" = "true" }
