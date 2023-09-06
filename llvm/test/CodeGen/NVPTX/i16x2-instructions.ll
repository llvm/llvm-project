; ## Support i16x2 instructions
; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90 -mattr=+ptx80 -asm-verbose=false \
; RUN:          -O0 -disable-post-ra -frame-pointer=all -verify-machineinstrs \
; RUN: | FileCheck -allow-deprecated-dag-overlap -check-prefixes CHECK,CHECK-I16x2 %s
; RUN: %if ptxas %{                                                           \
; RUN:   llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90 -asm-verbose=false \
; RUN:          -O0 -disable-post-ra -frame-pointer=all -verify-machineinstrs \
; RUN:   | %ptxas-verify -arch=sm_53                                          \
; RUN: %}
; ## No support for i16x2 instructions
; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_53 -asm-verbose=false \
; RUN:          -O0 -disable-post-ra -frame-pointer=all -verify-machineinstrs \
; RUN: | FileCheck -allow-deprecated-dag-overlap -check-prefixes CHECK,CHECK-NOI16x2 %s
; RUN: %if ptxas %{                                                           \
; RUN:   llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_53 -asm-verbose=false \
; RUN:          -O0 -disable-post-ra -frame-pointer=all -verify-machineinstrs \
; RUN:   | %ptxas-verify -arch=sm_53                                          \
; RUN: %}

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: test_ret_const(
; CHECK:     mov.u32         [[R:%r[0-9+]]], 131073;
; CHECK:     st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x i16> @test_ret_const() #0 {
  ret <2 x i16> <i16 1, i16 2>
}

; CHECK-LABEL: test_extract_0(
; CHECK:      ld.param.u32   [[A:%r[0-9]+]], [test_extract_0_param_0];
; CHECK:      mov.b32        {[[RS:%rs[0-9]+]], tmp}, [[A]];
; CHECK:      cvt.u32.u16    [[R:%r[0-9]+]], [[RS]];
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define i16 @test_extract_0(<2 x i16> %a) #0 {
  %e = extractelement <2 x i16> %a, i32 0
  ret i16 %e
}

; CHECK-LABEL: test_extract_1(
; CHECK:      ld.param.u32   [[A:%r[0-9]+]], [test_extract_1_param_0];
; CHECK:      mov.b32        {tmp, [[RS:%rs[0-9]+]]}, [[A]];
; CHECK:      cvt.u32.u16    [[R:%r[0-9]+]], [[RS]];
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define i16 @test_extract_1(<2 x i16> %a) #0 {
  %e = extractelement <2 x i16> %a, i32 1
  ret i16 %e
}

; CHECK-LABEL: test_extract_i(
; CHECK-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_extract_i_param_0];
; CHECK-DAG:  ld.param.u64    [[IDX:%rd[0-9]+]], [test_extract_i_param_1];
; CHECK-DAG:  setp.eq.s64     [[PRED:%p[0-9]+]], [[IDX]], 0;
; CHECK-DAG:  mov.b32         {[[E0:%rs[0-9]+]], [[E1:%rs[0-9]+]]}, [[A]];
; CHECK:      selp.b16        [[RS:%rs[0-9]+]], [[E0]], [[E1]], [[PRED]];
; CHECK:      cvt.u32.u16     [[R:%r[0-9]+]], [[RS]];
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define i16 @test_extract_i(<2 x i16> %a, i64 %idx) #0 {
  %e = extractelement <2 x i16> %a, i64 %idx
  ret i16 %e
}

; CHECK-LABEL: test_add(
; CHECK-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_add_param_0];
; CHECK-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_add_param_1];
;
; CHECK-I16x2-NEXT:  add.s16x2   [[R:%r[0-9]+]], [[A]], [[B]];
;
;	CHECK-NOI16x2-DAG: mov.b32 	{[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[A]];
;	CHECK-NOI16x2-DAG: mov.b32 	{[[RS2:%rs[0-9]+]], [[RS3:%rs[0-9]+]]}, [[B]];
;	CHECK-NOI16x2-DAG: add.s16 	[[RS4:%rs[0-9]+]], [[RS0]], [[RS2]];
;	CHECK-NOI16x2-DAG: add.s16 	[[RS5:%rs[0-9]+]], [[RS1]], [[RS3]];
;	CHECK-NOI16x2-DAG: mov.b32 	[[R:%r[0-9]+]], {[[RS4]], [[RS5]]};
;
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x i16> @test_add(<2 x i16> %a, <2 x i16> %b) #0 {
  %r = add <2 x i16> %a, %b
  ret <2 x i16> %r
}

; Check that we can lower add with immediate arguments.
; CHECK-LABEL: test_add_imm_0(
; CHECK-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_add_imm_0_param_0];
;
; CHECK-I16x2:        mov.u32        [[I:%r[0-9+]]], 131073;
; CHECK-I16x2:        add.s16x2      [[R:%r[0-9]+]], [[A]], [[I]];
;
;	CHECK-NOI16x2-DAG: mov.b32 	{[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[A]];
;	CHECK-NOI16x2-DAG: add.s16 	[[RS2:%rs[0-9]+]], [[RS0]], 1;
;	CHECK-NOI16x2-DAG: add.s16 	[[RS3:%rs[0-9]+]], [[RS1]], 2;
;	CHECK-NOI16x2-DAG: mov.b32 	[[R:%r[0-9]+]], {[[RS2]], [[RS3]]};
;
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x i16> @test_add_imm_0(<2 x i16> %a) #0 {
  %r = add <2 x i16> <i16 1, i16 2>, %a
  ret <2 x i16> %r
}

; CHECK-LABEL: test_add_imm_1(
; CHECK-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_add_imm_1_param_0];
;
; CHECK-I16x2:        mov.u32        [[I:%r[0-9+]]], 131073;
; CHECK-I16x2:        add.s16x2      [[R:%r[0-9]+]], [[A]], [[I]];
;
;	CHECK-NOI16x2-DAG: mov.b32 	{[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[A]];
;	CHECK-NOI16x2-DAG: add.s16 	[[RS2:%rs[0-9]+]], [[RS0]], 1;
;	CHECK-NOI16x2-DAG: add.s16 	[[RS3:%rs[0-9]+]], [[RS1]], 2;
;	CHECK-NOI16x2-DAG: mov.b32 	[[R:%r[0-9]+]], {[[RS2]], [[RS3]]};
;
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x i16> @test_add_imm_1(<2 x i16> %a) #0 {
  %r = add <2 x i16> %a, <i16 1, i16 2>
  ret <2 x i16> %r
}

; CHECK-LABEL: test_sub(
; CHECK-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_sub_param_0];
;
; CHECK-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_sub_param_1];
; CHECK-I16x2:   sub.s16x2   [[R:%r[0-9]+]], [[A]], [[B]];
;
;	CHECK-NOI16x2-DAG: mov.b32 	{[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[A]];
;	CHECK-NOI16x2-DAG: mov.b32 	{[[RS2:%rs[0-9]+]], [[RS3:%rs[0-9]+]]}, [[B]];
;	CHECK-NOI16x2-DAG: sub.s16 	[[RS4:%rs[0-9]+]], [[RS0]], [[RS2]];
;	CHECK-NOI16x2-DAG: sub.s16 	[[RS5:%rs[0-9]+]], [[RS1]], [[RS3]];
;	CHECK-NOI16x2-DAG: mov.b32 	[[R:%r[0-9]+]], {[[RS4]], [[RS5]]};
;
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x i16> @test_sub(<2 x i16> %a, <2 x i16> %b) #0 {
  %r = sub <2 x i16> %a, %b
  ret <2 x i16> %r
}

; CHECK-LABEL: test_smax(
; CHECK-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_smax_param_0];
;
; CHECK-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_smax_param_1];
; CHECK-I16x2:   max.s16x2   [[R:%r[0-9]+]], [[A]], [[B]];
;
;	CHECK-NOI16x2-DAG: mov.b32 	{[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[A]];
;	CHECK-NOI16x2-DAG: mov.b32 	{[[RS2:%rs[0-9]+]], [[RS3:%rs[0-9]+]]}, [[B]];
;	CHECK-NOI16x2-DAG: max.s16 	[[RS4:%rs[0-9]+]], [[RS0]], [[RS2]];
;	CHECK-NOI16x2-DAG: max.s16 	[[RS5:%rs[0-9]+]], [[RS1]], [[RS3]];
;	CHECK-NOI16x2-DAG: mov.b32 	[[R:%r[0-9]+]], {[[RS4]], [[RS5]]};
;
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x i16> @test_smax(<2 x i16> %a, <2 x i16> %b) #0 {
  %cmp = icmp sgt <2 x i16> %a, %b
  %r = select <2 x i1> %cmp, <2 x i16> %a, <2 x i16> %b
  ret <2 x i16> %r
}

; CHECK-LABEL: test_umax(
; CHECK-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_umax_param_0];
;
; CHECK-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_umax_param_1];
; CHECK-I16x2:   max.u16x2   [[R:%r[0-9]+]], [[A]], [[B]];
;
;	CHECK-NOI16x2-DAG: mov.b32 	{[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[A]];
;	CHECK-NOI16x2-DAG: mov.b32 	{[[RS2:%rs[0-9]+]], [[RS3:%rs[0-9]+]]}, [[B]];
;	CHECK-NOI16x2-DAG: max.u16 	[[RS4:%rs[0-9]+]], [[RS0]], [[RS2]];
;	CHECK-NOI16x2-DAG: max.u16 	[[RS5:%rs[0-9]+]], [[RS1]], [[RS3]];
;	CHECK-NOI16x2-DAG: mov.b32 	[[R:%r[0-9]+]], {[[RS4]], [[RS5]]};
;
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x i16> @test_umax(<2 x i16> %a, <2 x i16> %b) #0 {
  %cmp = icmp ugt <2 x i16> %a, %b
  %r = select <2 x i1> %cmp, <2 x i16> %a, <2 x i16> %b
  ret <2 x i16> %r
}

; CHECK-LABEL: test_smin(
; CHECK-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_smin_param_0];
;
; CHECK-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_smin_param_1];
; CHECK-I16x2:   min.s16x2   [[R:%r[0-9]+]], [[A]], [[B]];
;
;	CHECK-NOI16x2-DAG: mov.b32 	{[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[A]];
;	CHECK-NOI16x2-DAG: mov.b32 	{[[RS2:%rs[0-9]+]], [[RS3:%rs[0-9]+]]}, [[B]];
;	CHECK-NOI16x2-DAG: min.s16 	[[RS4:%rs[0-9]+]], [[RS0]], [[RS2]];
;	CHECK-NOI16x2-DAG: min.s16 	[[RS5:%rs[0-9]+]], [[RS1]], [[RS3]];
;	CHECK-NOI16x2-DAG: mov.b32 	[[R:%r[0-9]+]], {[[RS4]], [[RS5]]};
;
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x i16> @test_smin(<2 x i16> %a, <2 x i16> %b) #0 {
  %cmp = icmp sle <2 x i16> %a, %b
  %r = select <2 x i1> %cmp, <2 x i16> %a, <2 x i16> %b
  ret <2 x i16> %r
}

; CHECK-LABEL: test_umin(
; CHECK-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_umin_param_0];
;
; CHECK-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_umin_param_1];
; CHECK-I16x2:   min.u16x2   [[R:%r[0-9]+]], [[A]], [[B]];
;
;	CHECK-NOI16x2-DAG: mov.b32 	{[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[A]];
;	CHECK-NOI16x2-DAG: mov.b32 	{[[RS2:%rs[0-9]+]], [[RS3:%rs[0-9]+]]}, [[B]];
;	CHECK-NOI16x2-DAG: min.u16 	[[RS4:%rs[0-9]+]], [[RS0]], [[RS2]];
;	CHECK-NOI16x2-DAG: min.u16 	[[RS5:%rs[0-9]+]], [[RS1]], [[RS3]];
;	CHECK-NOI16x2-DAG: mov.b32 	[[R:%r[0-9]+]], {[[RS4]], [[RS5]]};
;
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x i16> @test_umin(<2 x i16> %a, <2 x i16> %b) #0 {
  %cmp = icmp ule <2 x i16> %a, %b
  %r = select <2 x i1> %cmp, <2 x i16> %a, <2 x i16> %b
  ret <2 x i16> %r
}

; CHECK-LABEL: test_mul(
; CHECK-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_mul_param_0];
; CHECK-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_mul_param_1];
;
;	CHECK-DAG: mov.b32 	    {[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[A]];
;	CHECK-DAG: mov.b32 	    {[[RS2:%rs[0-9]+]], [[RS3:%rs[0-9]+]]}, [[B]];
;	CHECK-DAG: mul.lo.s16 	[[RS4:%rs[0-9]+]], [[RS0]], [[RS2]];
;	CHECK-DAG: mul.lo.s16 	[[RS5:%rs[0-9]+]], [[RS1]], [[RS3]];
;	CHECK-DAG: mov.b32 	    [[R:%r[0-9]+]], {[[RS4]], [[RS5]]};
;
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x i16> @test_mul(<2 x i16> %a, <2 x i16> %b) #0 {
  %r = mul <2 x i16> %a, %b
  ret <2 x i16> %r
}


; CHECK-LABEL: .func test_ldst_v2i16(
; CHECK-DAG:    ld.param.u64    [[A:%rd[0-9]+]], [test_ldst_v2i16_param_0];
; CHECK-DAG:    ld.param.u64    [[B:%rd[0-9]+]], [test_ldst_v2i16_param_1];
; CHECK-DAG:    ld.u32          [[E:%r[0-9]+]], [[[A]]];
; CHECK-DAG:    st.u32          [[[B]]], [[E]];
; CHECK:        ret;
define void @test_ldst_v2i16(ptr %a, ptr %b) {
  %t1 = load <2 x i16>, ptr %a
  store <2 x i16> %t1, ptr %b, align 16
  ret void
}

; CHECK-LABEL: .func test_ldst_v3i16(
; CHECK-DAG:    ld.param.u64    %[[A:rd[0-9]+]], [test_ldst_v3i16_param_0];
; CHECK-DAG:    ld.param.u64    %[[B:rd[0-9]+]], [test_ldst_v3i16_param_1];
; -- v3 is inconvenient to capture as it's lowered as ld.b64 + fair
;    number of bitshifting instructions that may change at llvm's whim.
;    So we only verify that we only issue correct number of writes using
;    correct offset, but not the values we write.
; CHECK-DAG:    ld.u64
; CHECK-DAG:    st.u32          [%[[B]]],
; CHECK-DAG:    st.u16          [%[[B]]+4],
; CHECK:        ret;
define void @test_ldst_v3i16(ptr %a, ptr %b) {
  %t1 = load <3 x i16>, ptr %a
  store <3 x i16> %t1, ptr %b, align 16
  ret void
}

; CHECK-LABEL: .func test_ldst_v4i16(
; CHECK-DAG:    ld.param.u64    %[[A:rd[0-9]+]], [test_ldst_v4i16_param_0];
; CHECK-DAG:    ld.param.u64    %[[B:rd[0-9]+]], [test_ldst_v4i16_param_1];
; CHECK-DAG:    ld.v4.u16       {[[E0:%rs[0-9]+]], [[E1:%rs[0-9]+]], [[E2:%rs[0-9]+]], [[E3:%rs[0-9]+]]}, [%[[A]]];
; CHECK-DAG:    st.v4.u16       [%[[B]]], {[[E0]], [[E1]], [[E2]], [[E3]]};
; CHECK:        ret;
define void @test_ldst_v4i16(ptr %a, ptr %b) {
  %t1 = load <4 x i16>, ptr %a
  store <4 x i16> %t1, ptr %b, align 16
  ret void
}

; CHECK-LABEL: .func test_ldst_v8i16(
; CHECK-DAG:    ld.param.u64    %[[A:rd[0-9]+]], [test_ldst_v8i16_param_0];
; CHECK-DAG:    ld.param.u64    %[[B:rd[0-9]+]], [test_ldst_v8i16_param_1];
; CHECK-DAG:    ld.v4.b32       {[[E0:%r[0-9]+]], [[E1:%r[0-9]+]], [[E2:%r[0-9]+]], [[E3:%r[0-9]+]]}, [%[[A]]];
; CHECK-DAG:    st.v4.b32       [%[[B]]], {[[E0]], [[E1]], [[E2]], [[E3]]};
; CHECK:        ret;
define void @test_ldst_v8i16(ptr %a, ptr %b) {
  %t1 = load <8 x i16>, ptr %a
  store <8 x i16> %t1, ptr %b, align 16
  ret void
}

declare <2 x i16> @test_callee(<2 x i16> %a, <2 x i16> %b) #0

; CHECK-LABEL: test_call(
; CHECK-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_call_param_0];
; CHECK-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_call_param_1];
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
define <2 x i16> @test_call(<2 x i16> %a, <2 x i16> %b) #0 {
  %r = call <2 x i16> @test_callee(<2 x i16> %a, <2 x i16> %b)
  ret <2 x i16> %r
}

; CHECK-LABEL: test_call_flipped(
; CHECK-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_call_flipped_param_0];
; CHECK-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_call_flipped_param_1];
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
define <2 x i16> @test_call_flipped(<2 x i16> %a, <2 x i16> %b) #0 {
  %r = call <2 x i16> @test_callee(<2 x i16> %b, <2 x i16> %a)
  ret <2 x i16> %r
}

; CHECK-LABEL: test_tailcall_flipped(
; CHECK-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_tailcall_flipped_param_0];
; CHECK-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_tailcall_flipped_param_1];
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
define <2 x i16> @test_tailcall_flipped(<2 x i16> %a, <2 x i16> %b) #0 {
  %r = tail call <2 x i16> @test_callee(<2 x i16> %b, <2 x i16> %a)
  ret <2 x i16> %r
}

; CHECK-LABEL: test_select(
; CHECK-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_select_param_0];
; CHECK-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_select_param_1];
; CHECK-DAG:  ld.param.u8     [[C:%rs[0-9]+]], [test_select_param_2]
; CHECK-DAG:  setp.eq.b16     [[PRED:%p[0-9]+]], %rs{{.*}}, 1;
; CHECK-NEXT: selp.b32        [[R:%r[0-9]+]], [[A]], [[B]], [[PRED]];
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x i16> @test_select(<2 x i16> %a, <2 x i16> %b, i1 zeroext %c) #0 {
  %r = select i1 %c, <2 x i16> %a, <2 x i16> %b
  ret <2 x i16> %r
}

; CHECK-LABEL: test_select_cc(
; CHECK-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_select_cc_param_0];
; CHECK-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_select_cc_param_1];
; CHECK-DAG:  ld.param.u32    [[C:%r[0-9]+]], [test_select_cc_param_2];
; CHECK-DAG:  ld.param.u32    [[D:%r[0-9]+]], [test_select_cc_param_3];
; CHECK-DAG:  mov.b32        {[[C0:%rs[0-9]+]], [[C1:%rs[0-9]+]]}, [[C]]
; CHECK-DAG:  mov.b32        {[[D0:%rs[0-9]+]], [[D1:%rs[0-9]+]]}, [[D]]
; CHECK-DAG:  setp.ne.s16    [[P0:%p[0-9]+]], [[C0]], [[D0]]
; CHECK-DAG:  setp.ne.s16    [[P1:%p[0-9]+]], [[C1]], [[D1]]
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  mov.b32         {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-DAG:  selp.b16        [[R0:%rs[0-9]+]], [[A0]], [[B0]], [[P0]];
; CHECK-DAG:  selp.b16        [[R1:%rs[0-9]+]], [[A1]], [[B1]], [[P1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x i16> @test_select_cc(<2 x i16> %a, <2 x i16> %b, <2 x i16> %c, <2 x i16> %d) #0 {
  %cc = icmp ne <2 x i16> %c, %d
  %r = select <2 x i1> %cc, <2 x i16> %a, <2 x i16> %b
  ret <2 x i16> %r
}

; CHECK-LABEL: test_select_cc_i32_i16(
; CHECK-DAG:  ld.param.v2.u32    {[[A0:%r[0-9]+]], [[A1:%r[0-9]+]]}, [test_select_cc_i32_i16_param_0];
; CHECK-DAG:  ld.param.v2.u32    {[[B0:%r[0-9]+]], [[B1:%r[0-9]+]]}, [test_select_cc_i32_i16_param_1];
; CHECK-DAG:  ld.param.u32    [[C:%r[0-9]+]], [test_select_cc_i32_i16_param_2];
; CHECK-DAG:  ld.param.u32    [[D:%r[0-9]+]], [test_select_cc_i32_i16_param_3];
; CHECK-DAG: mov.b32         {[[C0:%rs[0-9]+]], [[C1:%rs[0-9]+]]}, [[C]]
; CHECK-DAG: mov.b32         {[[D0:%rs[0-9]+]], [[D1:%rs[0-9]+]]}, [[D]]
; CHECK-DAG: setp.ne.s16    [[P0:%p[0-9]+]], [[C0]], [[D0]]
; CHECK-DAG: setp.ne.s16    [[P1:%p[0-9]+]], [[C1]], [[D1]]
; CHECK-DAG: selp.b32        [[R0:%r[0-9]+]], [[A0]], [[B0]], [[P0]];
; CHECK-DAG: selp.b32        [[R1:%r[0-9]+]], [[A1]], [[B1]], [[P1]];
; CHECK-NEXT: st.param.v2.b32    [func_retval0+0], {[[R0]], [[R1]]};
; CHECK-NEXT: ret;
define <2 x i32> @test_select_cc_i32_i16(<2 x i32> %a, <2 x i32> %b,
                                           <2 x i16> %c, <2 x i16> %d) #0 {
  %cc = icmp ne <2 x i16> %c, %d
  %r = select <2 x i1> %cc, <2 x i32> %a, <2 x i32> %b
  ret <2 x i32> %r
}

; CHECK-LABEL: test_select_cc_i16_i32(
; CHECK-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_select_cc_i16_i32_param_0];
; CHECK-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_select_cc_i16_i32_param_1];
; CHECK-DAG:  ld.param.v2.u32 {[[C0:%r[0-9]+]], [[C1:%r[0-9]+]]}, [test_select_cc_i16_i32_param_2];
; CHECK-DAG:  ld.param.v2.u32 {[[D0:%r[0-9]+]], [[D1:%r[0-9]+]]}, [test_select_cc_i16_i32_param_3];
; CHECK-DAG:  setp.ne.s32    [[P0:%p[0-9]+]], [[C0]], [[D0]]
; CHECK-DAG:  setp.ne.s32    [[P1:%p[0-9]+]], [[C1]], [[D1]]
; CHECK-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  mov.b32         {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; CHECK-DAG:  selp.b16        [[R0:%rs[0-9]+]], [[A0]], [[B0]], [[P0]];
; CHECK-DAG:  selp.b16        [[R1:%rs[0-9]+]], [[A1]], [[B1]], [[P1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;
define <2 x i16> @test_select_cc_i16_i32(<2 x i16> %a, <2 x i16> %b,
                                          <2 x i32> %c, <2 x i32> %d) #0 {
  %cc = icmp ne <2 x i32> %c, %d
  %r = select <2 x i1> %cc, <2 x i16> %a, <2 x i16> %b
  ret <2 x i16> %r
}


; CHECK-LABEL: test_trunc_2xi32(
; CHECK:      ld.param.v2.u32 {[[A0:%r[0-9]+]], [[A1:%r[0-9]+]]}, [test_trunc_2xi32_param_0];
; CHECK-DAG:  cvt.u16.u32  [[R0:%rs[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.u16.u32  [[R1:%rs[0-9]+]], [[A1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x i16> @test_trunc_2xi32(<2 x i32> %a) #0 {
  %r = trunc <2 x i32> %a to <2 x i16>
  ret <2 x i16> %r
}

; CHECK-LABEL: test_trunc_2xi64(
; CHECK:      ld.param.v2.u64 {[[A0:%rd[0-9]+]], [[A1:%rd[0-9]+]]}, [test_trunc_2xi64_param_0];
; CHECK-DAG:  cvt.u16.u64  [[R0:%rs[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.u16.u64  [[R1:%rs[0-9]+]], [[A1]];
; CHECK:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;
define <2 x i16> @test_trunc_2xi64(<2 x i64> %a) #0 {
  %r = trunc <2 x i64> %a to <2 x i16>
  ret <2 x i16> %r
}

; CHECK-LABEL: test_zext_2xi32(
; CHECK:      ld.param.u32    [[A:%r[0-9]+]], [test_zext_2xi32_param_0];
; CHECK:      mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  cvt.u32.u16     [[R0:%r[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.u32.u16     [[R1:%r[0-9]+]], [[A1]];
; CHECK-NEXT: st.param.v2.b32 [func_retval0+0], {[[R0]], [[R1]]};
; CHECK:      ret;
define <2 x i32> @test_zext_2xi32(<2 x i16> %a) #0 {
  %r = zext <2 x i16> %a to <2 x i32>
  ret <2 x i32> %r
}

; CHECK-LABEL: test_zext_2xi64(
; CHECK:      ld.param.u32    [[A:%r[0-9]+]], [test_zext_2xi64_param_0];
; CHECK:      mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; CHECK-DAG:  cvt.u64.u16     [[R0:%rd[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.u64.u16     [[R1:%rd[0-9]+]], [[A1]];
; CHECK-NEXT: st.param.v2.b64 [func_retval0+0], {[[R0]], [[R1]]};
; CHECK:      ret;
define <2 x i64> @test_zext_2xi64(<2 x i16> %a) #0 {
  %r = zext <2 x i16> %a to <2 x i64>
  ret <2 x i64> %r
}

; CHECK-LABEL: test_bitcast_i32_to_2xi16(
; CHECK: ld.param.u32 	[[R:%r[0-9]+]], [test_bitcast_i32_to_2xi16_param_0];
; CHECK: st.param.b32 	[func_retval0+0], [[R]];
; CHECK: ret;
define <2 x i16> @test_bitcast_i32_to_2xi16(i32 %a) #0 {
  %r = bitcast i32 %a to <2 x i16>
  ret <2 x i16> %r
}

; CHECK-LABEL: test_bitcast_2xi16_to_i32(
; CHECK: ld.param.u32 	[[R:%r[0-9]+]], [test_bitcast_2xi16_to_i32_param_0];
; CHECK: st.param.b32 	[func_retval0+0], [[R]];
; CHECK: ret;
define i32 @test_bitcast_2xi16_to_i32(<2 x i16> %a) #0 {
  %r = bitcast <2 x i16> %a to i32
  ret i32 %r
}

; CHECK-LABEL: test_shufflevector(
; CHECK:	ld.param.u32 	[[R:%r[0-9]+]], [test_shufflevector_param_0];
; CHECK:	mov.b32 	{[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[R]];
; CHECK:	mov.b32 	[[R1:%r[0-9]+]], {[[RS1]], [[RS0]]};
; CHECK:	st.param.b32 	[func_retval0+0], [[R1]];
; CHECK:	ret;
define <2 x i16> @test_shufflevector(<2 x i16> %a) #0 {
  %s = shufflevector <2 x i16> %a, <2 x i16> undef, <2 x i32> <i32 1, i32 0>
  ret <2 x i16> %s
}

; CHECK-LABEL: test_insertelement(
; CHECK:  ld.param.u16 	[[B:%rs[0-9]+]], [test_insertelement_param_1];
; CHECK:	ld.param.u32 	[[A:%r[0-9]+]], [test_insertelement_param_0];
; CHECK:	{ .reg .b16 tmp; mov.b32 {[[R0:%rs[0-9]+]], tmp}, [[A]]; }
; CHECK:	mov.b32 	[[R1:%r[0-9]+]], {[[R0]], [[B]]};
; CHECK:	st.param.b32 	[func_retval0+0], [[R1]];
; CHECK:	ret;
define <2 x i16> @test_insertelement(<2 x i16> %a, i16 %x) #0 {
  %i = insertelement <2 x i16> %a, i16 %x, i64 1
  ret <2 x i16> %i
}

attributes #0 = { nounwind }
