; ## Support i16x2 instructions
; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90 -mattr=+ptx80 -asm-verbose=false \
; RUN:          -O0 -disable-post-ra -frame-pointer=all -verify-machineinstrs \
; RUN: | FileCheck -allow-deprecated-dag-overlap -check-prefixes COMMON,I16x2 %s
; RUN: %if ptxas %{                                                           \
; RUN:   llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90 -asm-verbose=false \
; RUN:          -O0 -disable-post-ra -frame-pointer=all -verify-machineinstrs \
; RUN:   | %ptxas-verify -arch=sm_90                                          \
; RUN: %}
; ## No support for i16x2 instructions
; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_53 -asm-verbose=false \
; RUN:          -O0 -disable-post-ra -frame-pointer=all -verify-machineinstrs \
; RUN: | FileCheck -allow-deprecated-dag-overlap -check-prefixes COMMON,NO-I16x2 %s
; RUN: %if ptxas %{                                                           \
; RUN:   llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_53 -asm-verbose=false \
; RUN:          -O0 -disable-post-ra -frame-pointer=all -verify-machineinstrs \
; RUN:   | %ptxas-verify -arch=sm_53                                          \
; RUN: %}

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; COMMON-LABEL: test_ret_const(
; COMMON:     mov.b32         [[R:%r[0-9+]]], 131073;
; COMMON:     st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_ret_const() #0 {
  ret <2 x i16> <i16 1, i16 2>
}

; COMMON-LABEL: test_extract_0(
; COMMON:      ld.param.u32   [[A:%r[0-9]+]], [test_extract_0_param_0];
; COMMON:      mov.b32        {[[RS:%rs[0-9]+]], tmp}, [[A]];
; COMMON:      cvt.u32.u16    [[R:%r[0-9]+]], [[RS]];
; COMMON:      st.param.b32    [func_retval0+0], [[R]];
; COMMON:      ret;
define i16 @test_extract_0(<2 x i16> %a) #0 {
  %e = extractelement <2 x i16> %a, i32 0
  ret i16 %e
}

; COMMON-LABEL: test_extract_1(
; COMMON:      ld.param.u32   [[A:%r[0-9]+]], [test_extract_1_param_0];
; COMMON:      mov.b32        {tmp, [[RS:%rs[0-9]+]]}, [[A]];
; COMMON:      cvt.u32.u16    [[R:%r[0-9]+]], [[RS]];
; COMMON:      st.param.b32    [func_retval0+0], [[R]];
; COMMON:      ret;
define i16 @test_extract_1(<2 x i16> %a) #0 {
  %e = extractelement <2 x i16> %a, i32 1
  ret i16 %e
}

; COMMON-LABEL: test_extract_i(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_extract_i_param_0];
; COMMON-DAG:  ld.param.u64    [[IDX:%rd[0-9]+]], [test_extract_i_param_1];
; COMMON-DAG:  setp.eq.s64     [[PRED:%p[0-9]+]], [[IDX]], 0;
; COMMON-DAG:  mov.b32         {[[E0:%rs[0-9]+]], [[E1:%rs[0-9]+]]}, [[A]];
; COMMON:      selp.b16        [[RS:%rs[0-9]+]], [[E0]], [[E1]], [[PRED]];
; COMMON:      cvt.u32.u16     [[R:%r[0-9]+]], [[RS]];
; COMMON:      st.param.b32    [func_retval0+0], [[R]];
; COMMON:      ret;
define i16 @test_extract_i(<2 x i16> %a, i64 %idx) #0 {
  %e = extractelement <2 x i16> %a, i64 %idx
  ret i16 %e
}

; COMMON-LABEL: test_add(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_add_param_0];
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_add_param_1];
;
; I16x2-NEXT:  add.s16x2   [[R:%r[0-9]+]], [[A]], [[B]];
;
; NO-I16x2-DAG: mov.b32 	{[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[A]];
; NO-I16x2-DAG: mov.b32 	{[[RS2:%rs[0-9]+]], [[RS3:%rs[0-9]+]]}, [[B]];
; NO-I16x2-DAG: add.s16 	[[RS4:%rs[0-9]+]], [[RS0]], [[RS2]];
; NO-I16x2-DAG: add.s16 	[[RS5:%rs[0-9]+]], [[RS1]], [[RS3]];
; NO-I16x2-DAG: mov.b32 	[[R:%r[0-9]+]], {[[RS4]], [[RS5]]};
;
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_add(<2 x i16> %a, <2 x i16> %b) #0 {
  %r = add <2 x i16> %a, %b
  ret <2 x i16> %r
}

; Check that we can lower add with immediate arguments.
; COMMON-LABEL: test_add_imm_0(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_add_imm_0_param_0];
;
; I16x2:        mov.b32        [[I:%r[0-9+]]], 131073;
; I16x2:        add.s16x2      [[R:%r[0-9]+]], [[A]], [[I]];
;
;	NO-I16x2-DAG: mov.b32 	{[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[A]];
;	NO-I16x2-DAG: add.s16 	[[RS2:%rs[0-9]+]], [[RS0]], 1;
;	NO-I16x2-DAG: add.s16 	[[RS3:%rs[0-9]+]], [[RS1]], 2;
;	NO-I16x2-DAG: mov.b32 	[[R:%r[0-9]+]], {[[RS2]], [[RS3]]};
;
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_add_imm_0(<2 x i16> %a) #0 {
  %r = add <2 x i16> <i16 1, i16 2>, %a
  ret <2 x i16> %r
}

; COMMON-LABEL: test_add_imm_1(
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_add_imm_1_param_0];
;
; I16x2:        mov.b32        [[I:%r[0-9+]]], 131073;
; I16x2:        add.s16x2      [[R:%r[0-9]+]], [[A]], [[I]];
;
;	NO-I16x2-DAG: mov.b32 	{[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[A]];
;	NO-I16x2-DAG: add.s16 	[[RS2:%rs[0-9]+]], [[RS0]], 1;
;	NO-I16x2-DAG: add.s16 	[[RS3:%rs[0-9]+]], [[RS1]], 2;
;	NO-I16x2-DAG: mov.b32 	[[R:%r[0-9]+]], {[[RS2]], [[RS3]]};
;
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_add_imm_1(<2 x i16> %a) #0 {
  %r = add <2 x i16> %a, <i16 1, i16 2>
  ret <2 x i16> %r
}

; COMMON-LABEL: test_sub(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_sub_param_0];
;
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_sub_param_1];
;
; COMMON-DAG:  mov.b32 	{[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[A]];
; COMMON-DAG:  mov.b32 	{[[RS2:%rs[0-9]+]], [[RS3:%rs[0-9]+]]}, [[B]];
; COMMON-DAG:  sub.s16 	[[RS4:%rs[0-9]+]], [[RS0]], [[RS2]];
; COMMON-DAG:  sub.s16 	[[RS5:%rs[0-9]+]], [[RS1]], [[RS3]];
; COMMON-DAG:  mov.b32 	[[R:%r[0-9]+]], {[[RS4]], [[RS5]]};
;
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_sub(<2 x i16> %a, <2 x i16> %b) #0 {
  %r = sub <2 x i16> %a, %b
  ret <2 x i16> %r
}

; COMMON-LABEL: test_smax(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_smax_param_0];
;
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_smax_param_1];
; I16x2:   max.s16x2   [[R:%r[0-9]+]], [[A]], [[B]];
;
;	NO-I16x2-DAG: mov.b32 	{[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[A]];
;	NO-I16x2-DAG: mov.b32 	{[[RS2:%rs[0-9]+]], [[RS3:%rs[0-9]+]]}, [[B]];
;	NO-I16x2-DAG: max.s16 	[[RS4:%rs[0-9]+]], [[RS0]], [[RS2]];
;	NO-I16x2-DAG: max.s16 	[[RS5:%rs[0-9]+]], [[RS1]], [[RS3]];
;	NO-I16x2-DAG: mov.b32 	[[R:%r[0-9]+]], {[[RS4]], [[RS5]]};
;
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_smax(<2 x i16> %a, <2 x i16> %b) #0 {
  %cmp = icmp sgt <2 x i16> %a, %b
  %r = select <2 x i1> %cmp, <2 x i16> %a, <2 x i16> %b
  ret <2 x i16> %r
}

; COMMON-LABEL: test_umax(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_umax_param_0];
;
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_umax_param_1];
; I16x2:   max.u16x2   [[R:%r[0-9]+]], [[A]], [[B]];
;
;	NO-I16x2-DAG: mov.b32 	{[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[A]];
;	NO-I16x2-DAG: mov.b32 	{[[RS2:%rs[0-9]+]], [[RS3:%rs[0-9]+]]}, [[B]];
;	NO-I16x2-DAG: max.u16 	[[RS4:%rs[0-9]+]], [[RS0]], [[RS2]];
;	NO-I16x2-DAG: max.u16 	[[RS5:%rs[0-9]+]], [[RS1]], [[RS3]];
;	NO-I16x2-DAG: mov.b32 	[[R:%r[0-9]+]], {[[RS4]], [[RS5]]};
;
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_umax(<2 x i16> %a, <2 x i16> %b) #0 {
  %cmp = icmp ugt <2 x i16> %a, %b
  %r = select <2 x i1> %cmp, <2 x i16> %a, <2 x i16> %b
  ret <2 x i16> %r
}

; COMMON-LABEL: test_smin(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_smin_param_0];
;
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_smin_param_1];
; I16x2:   min.s16x2   [[R:%r[0-9]+]], [[A]], [[B]];
;
;	NO-I16x2-DAG: mov.b32 	{[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[A]];
;	NO-I16x2-DAG: mov.b32 	{[[RS2:%rs[0-9]+]], [[RS3:%rs[0-9]+]]}, [[B]];
;	NO-I16x2-DAG: min.s16 	[[RS4:%rs[0-9]+]], [[RS0]], [[RS2]];
;	NO-I16x2-DAG: min.s16 	[[RS5:%rs[0-9]+]], [[RS1]], [[RS3]];
;	NO-I16x2-DAG: mov.b32 	[[R:%r[0-9]+]], {[[RS4]], [[RS5]]};
;
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_smin(<2 x i16> %a, <2 x i16> %b) #0 {
  %cmp = icmp sle <2 x i16> %a, %b
  %r = select <2 x i1> %cmp, <2 x i16> %a, <2 x i16> %b
  ret <2 x i16> %r
}

; COMMON-LABEL: test_umin(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_umin_param_0];
;
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_umin_param_1];
; I16x2:   min.u16x2   [[R:%r[0-9]+]], [[A]], [[B]];
;
;	NO-I16x2-DAG: mov.b32 	{[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[A]];
;	NO-I16x2-DAG: mov.b32 	{[[RS2:%rs[0-9]+]], [[RS3:%rs[0-9]+]]}, [[B]];
;	NO-I16x2-DAG: min.u16 	[[RS4:%rs[0-9]+]], [[RS0]], [[RS2]];
;	NO-I16x2-DAG: min.u16 	[[RS5:%rs[0-9]+]], [[RS1]], [[RS3]];
;	NO-I16x2-DAG: mov.b32 	[[R:%r[0-9]+]], {[[RS4]], [[RS5]]};
;
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_umin(<2 x i16> %a, <2 x i16> %b) #0 {
  %cmp = icmp ule <2 x i16> %a, %b
  %r = select <2 x i1> %cmp, <2 x i16> %a, <2 x i16> %b
  ret <2 x i16> %r
}

; COMMON-LABEL: test_mul(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_mul_param_0];
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_mul_param_1];
;
;	COMMON-DAG: mov.b32 	    {[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[A]];
;	COMMON-DAG: mov.b32 	    {[[RS2:%rs[0-9]+]], [[RS3:%rs[0-9]+]]}, [[B]];
;	COMMON-DAG: mul.lo.s16 	[[RS4:%rs[0-9]+]], [[RS0]], [[RS2]];
;	COMMON-DAG: mul.lo.s16 	[[RS5:%rs[0-9]+]], [[RS1]], [[RS3]];
;	COMMON-DAG: mov.b32 	    [[R:%r[0-9]+]], {[[RS4]], [[RS5]]};
;
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_mul(<2 x i16> %a, <2 x i16> %b) #0 {
  %r = mul <2 x i16> %a, %b
  ret <2 x i16> %r
}

;; Logical ops are available on all GPUs as regular 32-bit logical ops
; COMMON-LABEL: test_or(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_or_param_0];
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_or_param_1];
; COMMON-NEXT: or.b32          [[R:%r[0-9]+]], [[A]], [[B]];
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_or(<2 x i16> %a, <2 x i16> %b) #0 {
  %r = or <2 x i16> %a, %b
  ret <2 x i16> %r
}

; Ops that operate on computed arguments go though a different lowering path.
; compared to the ones that operate on loaded data. So we test them separately.
; COMMON-LABEL: test_or_computed(
; COMMON:        ld.param.u16    [[A:%rs[0-9+]]], [test_or_computed_param_0];
; COMMON-DAG:    mov.u16         [[C0:%rs[0-9]+]], 0;
; COMMON-DAG:    mov.b32         [[R1:%r[0-9]+]], {[[A]], [[C0]]};
; COMMON-DAG:    mov.u16         [[C5:%rs[0-9]+]], 5;
; COMMON-DAG:    mov.b32         [[R2:%r[0-9]+]], {[[A]], [[C5]]};
; COMMON:        or.b32          [[R:%r[0-9]+]], [[R2]], [[R1]];
; COMMON-NEXT:   st.param.b32    [func_retval0+0], [[R]];
define <2 x i16> @test_or_computed(i16 %a) {
  %ins.0 = insertelement <2 x i16> zeroinitializer, i16 %a, i32 0
  %ins.1 = insertelement <2 x i16> %ins.0, i16 5, i32 1
  %r = or <2 x i16> %ins.1, %ins.0
  ret <2 x i16> %r
}

; Check that we can lower or with immediate arguments.
; COMMON-LABEL: test_or_imm_0(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_or_imm_0_param_0];
; COMMON-NEXT: or.b32          [[R:%r[0-9]+]], [[A]], 131073;
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_or_imm_0(<2 x i16> %a) #0 {
  %r = or <2 x i16> <i16 1, i16 2>, %a
  ret <2 x i16> %r
}

; COMMON-LABEL: test_or_imm_1(
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_or_imm_1_param_0];
; COMMON-NEXT: or.b32          [[R:%r[0-9]+]], [[A]], 131073;
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_or_imm_1(<2 x i16> %a) #0 {
  %r = or <2 x i16> %a, <i16 1, i16 2>
  ret <2 x i16> %r
}

; COMMON-LABEL: test_xor(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_xor_param_0];
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_xor_param_1];
; COMMON-NEXT: xor.b32         [[R:%r[0-9]+]], [[A]], [[B]];
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_xor(<2 x i16> %a, <2 x i16> %b) #0 {
  %r = xor <2 x i16> %a, %b
  ret <2 x i16> %r
}

; COMMON-LABEL: test_xor_computed(
; COMMON:        ld.param.u16    [[A:%rs[0-9+]]], [test_xor_computed_param_0];
; COMMON-DAG:    mov.u16         [[C0:%rs[0-9]+]], 0;
; COMMON-DAG:    mov.b32         [[R1:%r[0-9]+]], {[[A]], [[C0]]};
; COMMON-DAG:    mov.u16         [[C5:%rs[0-9]+]], 5;
; COMMON-DAG:    mov.b32         [[R2:%r[0-9]+]], {[[A]], [[C5]]};
; COMMON:        xor.b32         [[R:%r[0-9]+]], [[R2]], [[R1]];
; COMMON-NEXT:   st.param.b32    [func_retval0+0], [[R]];
define <2 x i16> @test_xor_computed(i16 %a) {
  %ins.0 = insertelement <2 x i16> zeroinitializer, i16 %a, i32 0
  %ins.1 = insertelement <2 x i16> %ins.0, i16 5, i32 1
  %r = xor <2 x i16> %ins.1, %ins.0
  ret <2 x i16> %r
}

; Check that we can lower xor with immediate arguments.
; COMMON-LABEL: test_xor_imm_0(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_xor_imm_0_param_0];
; COMMON-NEXT: xor.b32         [[R:%r[0-9]+]], [[A]], 131073;
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_xor_imm_0(<2 x i16> %a) #0 {
  %r = xor <2 x i16> <i16 1, i16 2>, %a
  ret <2 x i16> %r
}

; COMMON-LABEL: test_xor_imm_1(
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_xor_imm_1_param_0];
; COMMON-NEXT: xor.b32         [[R:%r[0-9]+]], [[A]], 131073;
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_xor_imm_1(<2 x i16> %a) #0 {
  %r = xor <2 x i16> %a, <i16 1, i16 2>
  ret <2 x i16> %r
}

; COMMON-LABEL: test_and(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_and_param_0];
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_and_param_1];
; COMMON-NEXT: and.b32          [[R:%r[0-9]+]], [[A]], [[B]];
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_and(<2 x i16> %a, <2 x i16> %b) #0 {
  %r = and <2 x i16> %a, %b
  ret <2 x i16> %r
}

; Ops that operate on computed arguments go though a different lowering path.
; compared to the ones that operate on loaded data. So we test them separately.
; COMMON-LABEL: test_and_computed(
; COMMON:        ld.param.u16    [[A:%rs[0-9+]]], [test_and_computed_param_0];
; COMMON-DAG:    mov.u16         [[C0:%rs[0-9]+]], 0;
; COMMON-DAG:    mov.b32         [[R1:%r[0-9]+]], {[[A]], [[C0]]};
; COMMON-DAG:    mov.u16         [[C5:%rs[0-9]+]], 5;
; COMMON-DAG:    mov.b32         [[R2:%r[0-9]+]], {[[A]], [[C5]]};
; COMMON:        and.b32          [[R:%r[0-9]+]], [[R2]], [[R1]];
; COMMON-NEXT:   st.param.b32    [func_retval0+0], [[R]];
define <2 x i16> @test_and_computed(i16 %a) {
  %ins.0 = insertelement <2 x i16> zeroinitializer, i16 %a, i32 0
  %ins.1 = insertelement <2 x i16> %ins.0, i16 5, i32 1
  %r = and <2 x i16> %ins.1, %ins.0
  ret <2 x i16> %r
}

; Check that we can lower and with immediate arguments.
; COMMON-LABEL: test_and_imm_0(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_and_imm_0_param_0];
; COMMON-NEXT: and.b32          [[R:%r[0-9]+]], [[A]], 131073;
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_and_imm_0(<2 x i16> %a) #0 {
  %r = and <2 x i16> <i16 1, i16 2>, %a
  ret <2 x i16> %r
}

; COMMON-LABEL: test_and_imm_1(
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_and_imm_1_param_0];
; COMMON-NEXT: and.b32          [[R:%r[0-9]+]], [[A]], 131073;
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_and_imm_1(<2 x i16> %a) #0 {
  %r = and <2 x i16> %a, <i16 1, i16 2>
  ret <2 x i16> %r
}

; COMMON-LABEL: .func test_ldst_v2i16(
; COMMON-DAG:    ld.param.u64    [[A:%rd[0-9]+]], [test_ldst_v2i16_param_0];
; COMMON-DAG:    ld.param.u64    [[B:%rd[0-9]+]], [test_ldst_v2i16_param_1];
; COMMON-DAG:    ld.u32          [[E:%r[0-9]+]], [[[A]]];
; COMMON-DAG:    st.u32          [[[B]]], [[E]];
; COMMON:        ret;
define void @test_ldst_v2i16(ptr %a, ptr %b) {
  %t1 = load <2 x i16>, ptr %a
  store <2 x i16> %t1, ptr %b, align 16
  ret void
}

; COMMON-LABEL: .func test_ldst_v3i16(
; COMMON-DAG:    ld.param.u64    %[[A:rd[0-9]+]], [test_ldst_v3i16_param_0];
; COMMON-DAG:    ld.param.u64    %[[B:rd[0-9]+]], [test_ldst_v3i16_param_1];
; -- v3 is inconvenient to capture as it's lowered as ld.b64 + fair
;    number of bitshifting instructions that may change at llvm's whim.
;    So we only verify that we only issue correct number of writes using
;    correct offset, but not the values we write.
; COMMON-DAG:    ld.u64
; COMMON-DAG:    st.u32          [%[[B]]],
; COMMON-DAG:    st.u16          [%[[B]]+4],
; COMMON:        ret;
define void @test_ldst_v3i16(ptr %a, ptr %b) {
  %t1 = load <3 x i16>, ptr %a
  store <3 x i16> %t1, ptr %b, align 16
  ret void
}

; COMMON-LABEL: .func test_ldst_v4i16(
; COMMON-DAG:    ld.param.u64    %[[A:rd[0-9]+]], [test_ldst_v4i16_param_0];
; COMMON-DAG:    ld.param.u64    %[[B:rd[0-9]+]], [test_ldst_v4i16_param_1];
; COMMON-DAG:    ld.v4.u16       {[[E0:%rs[0-9]+]], [[E1:%rs[0-9]+]], [[E2:%rs[0-9]+]], [[E3:%rs[0-9]+]]}, [%[[A]]];
; COMMON-DAG:    st.v4.u16       [%[[B]]], {[[E0]], [[E1]], [[E2]], [[E3]]};
; COMMON:        ret;
define void @test_ldst_v4i16(ptr %a, ptr %b) {
  %t1 = load <4 x i16>, ptr %a
  store <4 x i16> %t1, ptr %b, align 16
  ret void
}

; COMMON-LABEL: .func test_ldst_v8i16(
; COMMON-DAG:    ld.param.u64    %[[A:rd[0-9]+]], [test_ldst_v8i16_param_0];
; COMMON-DAG:    ld.param.u64    %[[B:rd[0-9]+]], [test_ldst_v8i16_param_1];
; COMMON-DAG:    ld.v4.b32       {[[E0:%r[0-9]+]], [[E1:%r[0-9]+]], [[E2:%r[0-9]+]], [[E3:%r[0-9]+]]}, [%[[A]]];
; COMMON-DAG:    st.v4.b32       [%[[B]]], {[[E0]], [[E1]], [[E2]], [[E3]]};
; COMMON:        ret;
define void @test_ldst_v8i16(ptr %a, ptr %b) {
  %t1 = load <8 x i16>, ptr %a
  store <8 x i16> %t1, ptr %b, align 16
  ret void
}

declare <2 x i16> @test_callee(<2 x i16> %a, <2 x i16> %b) #0

; COMMON-LABEL: test_call(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_call_param_0];
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_call_param_1];
; COMMON:      {
; COMMON-DAG:  .param .align 4 .b8 param0[4];
; COMMON-DAG:  .param .align 4 .b8 param1[4];
; COMMON-DAG:  st.param.b32    [param0+0], [[A]];
; COMMON-DAG:  st.param.b32    [param1+0], [[B]];
; COMMON-DAG:  .param .align 4 .b8 retval0[4];
; COMMON:      call.uni (retval0),
; COMMON-NEXT:        test_callee,
; COMMON:      );
; COMMON-NEXT: ld.param.b32    [[R:%r[0-9]+]], [retval0+0];
; COMMON-NEXT: }
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_call(<2 x i16> %a, <2 x i16> %b) #0 {
  %r = call <2 x i16> @test_callee(<2 x i16> %a, <2 x i16> %b)
  ret <2 x i16> %r
}

; COMMON-LABEL: test_call_flipped(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_call_flipped_param_0];
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_call_flipped_param_1];
; COMMON:      {
; COMMON-DAG:  .param .align 4 .b8 param0[4];
; COMMON-DAG:  .param .align 4 .b8 param1[4];
; COMMON-DAG:  st.param.b32    [param0+0], [[B]];
; COMMON-DAG:  st.param.b32    [param1+0], [[A]];
; COMMON-DAG:  .param .align 4 .b8 retval0[4];
; COMMON:      call.uni (retval0),
; COMMON-NEXT:        test_callee,
; COMMON:      );
; COMMON-NEXT: ld.param.b32    [[R:%r[0-9]+]], [retval0+0];
; COMMON-NEXT: }
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_call_flipped(<2 x i16> %a, <2 x i16> %b) #0 {
  %r = call <2 x i16> @test_callee(<2 x i16> %b, <2 x i16> %a)
  ret <2 x i16> %r
}

; COMMON-LABEL: test_tailcall_flipped(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_tailcall_flipped_param_0];
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_tailcall_flipped_param_1];
; COMMON:      {
; COMMON-DAG:  .param .align 4 .b8 param0[4];
; COMMON-DAG:  .param .align 4 .b8 param1[4];
; COMMON-DAG:  st.param.b32    [param0+0], [[B]];
; COMMON-DAG:  st.param.b32    [param1+0], [[A]];
; COMMON-DAG:  .param .align 4 .b8 retval0[4];
; COMMON:      call.uni (retval0),
; COMMON-NEXT:        test_callee,
; COMMON:      );
; COMMON-NEXT: ld.param.b32    [[R:%r[0-9]+]], [retval0+0];
; COMMON-NEXT: }
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_tailcall_flipped(<2 x i16> %a, <2 x i16> %b) #0 {
  %r = tail call <2 x i16> @test_callee(<2 x i16> %b, <2 x i16> %a)
  ret <2 x i16> %r
}

; COMMON-LABEL: test_select(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_select_param_0];
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_select_param_1];
; COMMON-DAG:  ld.param.u8     [[C:%rs[0-9]+]], [test_select_param_2]
; COMMON-DAG:  setp.eq.b16     [[PRED:%p[0-9]+]], %rs{{.*}}, 1;
; COMMON-NEXT: selp.b32        [[R:%r[0-9]+]], [[A]], [[B]], [[PRED]];
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_select(<2 x i16> %a, <2 x i16> %b, i1 zeroext %c) #0 {
  %r = select i1 %c, <2 x i16> %a, <2 x i16> %b
  ret <2 x i16> %r
}

; COMMON-LABEL: test_select_cc(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_select_cc_param_0];
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_select_cc_param_1];
; COMMON-DAG:  ld.param.u32    [[C:%r[0-9]+]], [test_select_cc_param_2];
; COMMON-DAG:  ld.param.u32    [[D:%r[0-9]+]], [test_select_cc_param_3];
; COMMON-DAG:  mov.b32        {[[C0:%rs[0-9]+]], [[C1:%rs[0-9]+]]}, [[C]]
; COMMON-DAG:  mov.b32        {[[D0:%rs[0-9]+]], [[D1:%rs[0-9]+]]}, [[D]]
; COMMON-DAG:  setp.ne.s16    [[P0:%p[0-9]+]], [[C0]], [[D0]]
; COMMON-DAG:  setp.ne.s16    [[P1:%p[0-9]+]], [[C1]], [[D1]]
; COMMON-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; COMMON-DAG:  mov.b32         {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; COMMON-DAG:  selp.b16        [[R0:%rs[0-9]+]], [[A0]], [[B0]], [[P0]];
; COMMON-DAG:  selp.b16        [[R1:%rs[0-9]+]], [[A1]], [[B1]], [[P1]];
; COMMON:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_select_cc(<2 x i16> %a, <2 x i16> %b, <2 x i16> %c, <2 x i16> %d) #0 {
  %cc = icmp ne <2 x i16> %c, %d
  %r = select <2 x i1> %cc, <2 x i16> %a, <2 x i16> %b
  ret <2 x i16> %r
}

; COMMON-LABEL: test_select_cc_i32_i16(
; COMMON-DAG:  ld.param.v2.u32    {[[A0:%r[0-9]+]], [[A1:%r[0-9]+]]}, [test_select_cc_i32_i16_param_0];
; COMMON-DAG:  ld.param.v2.u32    {[[B0:%r[0-9]+]], [[B1:%r[0-9]+]]}, [test_select_cc_i32_i16_param_1];
; COMMON-DAG:  ld.param.u32    [[C:%r[0-9]+]], [test_select_cc_i32_i16_param_2];
; COMMON-DAG:  ld.param.u32    [[D:%r[0-9]+]], [test_select_cc_i32_i16_param_3];
; COMMON-DAG: mov.b32         {[[C0:%rs[0-9]+]], [[C1:%rs[0-9]+]]}, [[C]]
; COMMON-DAG: mov.b32         {[[D0:%rs[0-9]+]], [[D1:%rs[0-9]+]]}, [[D]]
; COMMON-DAG: setp.ne.s16    [[P0:%p[0-9]+]], [[C0]], [[D0]]
; COMMON-DAG: setp.ne.s16    [[P1:%p[0-9]+]], [[C1]], [[D1]]
; COMMON-DAG: selp.b32        [[R0:%r[0-9]+]], [[A0]], [[B0]], [[P0]];
; COMMON-DAG: selp.b32        [[R1:%r[0-9]+]], [[A1]], [[B1]], [[P1]];
; COMMON-NEXT: st.param.v2.b32    [func_retval0+0], {[[R0]], [[R1]]};
; COMMON-NEXT: ret;
define <2 x i32> @test_select_cc_i32_i16(<2 x i32> %a, <2 x i32> %b,
                                           <2 x i16> %c, <2 x i16> %d) #0 {
  %cc = icmp ne <2 x i16> %c, %d
  %r = select <2 x i1> %cc, <2 x i32> %a, <2 x i32> %b
  ret <2 x i32> %r
}

; COMMON-LABEL: test_select_cc_i16_i32(
; COMMON-DAG:  ld.param.u32    [[A:%r[0-9]+]], [test_select_cc_i16_i32_param_0];
; COMMON-DAG:  ld.param.u32    [[B:%r[0-9]+]], [test_select_cc_i16_i32_param_1];
; COMMON-DAG:  ld.param.v2.u32 {[[C0:%r[0-9]+]], [[C1:%r[0-9]+]]}, [test_select_cc_i16_i32_param_2];
; COMMON-DAG:  ld.param.v2.u32 {[[D0:%r[0-9]+]], [[D1:%r[0-9]+]]}, [test_select_cc_i16_i32_param_3];
; COMMON-DAG:  setp.ne.s32    [[P0:%p[0-9]+]], [[C0]], [[D0]]
; COMMON-DAG:  setp.ne.s32    [[P1:%p[0-9]+]], [[C1]], [[D1]]
; COMMON-DAG:  mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; COMMON-DAG:  mov.b32         {[[B0:%rs[0-9]+]], [[B1:%rs[0-9]+]]}, [[B]]
; COMMON-DAG:  selp.b16        [[R0:%rs[0-9]+]], [[A0]], [[B0]], [[P0]];
; COMMON-DAG:  selp.b16        [[R1:%rs[0-9]+]], [[A1]], [[B1]], [[P1]];
; COMMON:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; COMMON-NEXT: st.param.b32    [func_retval0+0], [[R]];
; COMMON-NEXT: ret;
define <2 x i16> @test_select_cc_i16_i32(<2 x i16> %a, <2 x i16> %b,
                                          <2 x i32> %c, <2 x i32> %d) #0 {
  %cc = icmp ne <2 x i32> %c, %d
  %r = select <2 x i1> %cc, <2 x i16> %a, <2 x i16> %b
  ret <2 x i16> %r
}


; COMMON-LABEL: test_trunc_2xi32(
; COMMON:      ld.param.v2.u32 {[[A0:%r[0-9]+]], [[A1:%r[0-9]+]]}, [test_trunc_2xi32_param_0];
; COMMON-DAG:  cvt.u16.u32  [[R0:%rs[0-9]+]], [[A0]];
; COMMON-DAG:  cvt.u16.u32  [[R1:%rs[0-9]+]], [[A1]];
; COMMON:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; COMMON:      st.param.b32    [func_retval0+0], [[R]];
; COMMON:      ret;
define <2 x i16> @test_trunc_2xi32(<2 x i32> %a) #0 {
  %r = trunc <2 x i32> %a to <2 x i16>
  ret <2 x i16> %r
}

; COMMON-LABEL: test_trunc_2xi64(
; COMMON:      ld.param.v2.u64 {[[A0:%rd[0-9]+]], [[A1:%rd[0-9]+]]}, [test_trunc_2xi64_param_0];
; COMMON-DAG:  cvt.u16.u64  [[R0:%rs[0-9]+]], [[A0]];
; COMMON-DAG:  cvt.u16.u64  [[R1:%rs[0-9]+]], [[A1]];
; COMMON:      mov.b32         [[R:%r[0-9]+]], {[[R0]], [[R1]]}
; COMMON:      st.param.b32    [func_retval0+0], [[R]];
; COMMON:      ret;
define <2 x i16> @test_trunc_2xi64(<2 x i64> %a) #0 {
  %r = trunc <2 x i64> %a to <2 x i16>
  ret <2 x i16> %r
}

; COMMON-LABEL: test_zext_2xi32(
; COMMON:      ld.param.u32    [[A:%r[0-9]+]], [test_zext_2xi32_param_0];
; COMMON:      mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; COMMON-DAG:  cvt.u32.u16     [[R0:%r[0-9]+]], [[A0]];
; COMMON-DAG:  cvt.u32.u16     [[R1:%r[0-9]+]], [[A1]];
; COMMON-NEXT: st.param.v2.b32 [func_retval0+0], {[[R0]], [[R1]]};
; COMMON:      ret;
define <2 x i32> @test_zext_2xi32(<2 x i16> %a) #0 {
  %r = zext <2 x i16> %a to <2 x i32>
  ret <2 x i32> %r
}

; COMMON-LABEL: test_zext_2xi64(
; COMMON:      ld.param.u32    [[A:%r[0-9]+]], [test_zext_2xi64_param_0];
; COMMON:      mov.b32         {[[A0:%rs[0-9]+]], [[A1:%rs[0-9]+]]}, [[A]]
; COMMON-DAG:  cvt.u64.u16     [[R0:%rd[0-9]+]], [[A0]];
; COMMON-DAG:  cvt.u64.u16     [[R1:%rd[0-9]+]], [[A1]];
; COMMON-NEXT: st.param.v2.b64 [func_retval0+0], {[[R0]], [[R1]]};
; COMMON:      ret;
define <2 x i64> @test_zext_2xi64(<2 x i16> %a) #0 {
  %r = zext <2 x i16> %a to <2 x i64>
  ret <2 x i64> %r
}

; COMMON-LABEL: test_bitcast_i32_to_2xi16(
; COMMON: ld.param.u32 	[[R:%r[0-9]+]], [test_bitcast_i32_to_2xi16_param_0];
; COMMON: st.param.b32 	[func_retval0+0], [[R]];
; COMMON: ret;
define <2 x i16> @test_bitcast_i32_to_2xi16(i32 %a) #0 {
  %r = bitcast i32 %a to <2 x i16>
  ret <2 x i16> %r
}

; COMMON-LABEL: test_bitcast_2xi16_to_i32(
; COMMON: ld.param.u32 	[[R:%r[0-9]+]], [test_bitcast_2xi16_to_i32_param_0];
; COMMON: st.param.b32 	[func_retval0+0], [[R]];
; COMMON: ret;
define i32 @test_bitcast_2xi16_to_i32(<2 x i16> %a) #0 {
  %r = bitcast <2 x i16> %a to i32
  ret i32 %r
}

; COMMON-LABEL: test_bitcast_2xi16_to_2xhalf(
; COMMON: ld.param.u16 	[[RS1:%rs[0-9]+]], [test_bitcast_2xi16_to_2xhalf_param_0];
; COMMON:	mov.u16 	[[RS2:%rs[0-9]+]], 5;
; COMMON:	mov.b32 	[[R:%r[0-9]+]], {[[RS1]], [[RS2]]};
; COMMON: st.param.b32 	[func_retval0+0], [[R]];
; COMMON: ret;
define <2 x half> @test_bitcast_2xi16_to_2xhalf(i16 %a) #0 {
  %ins.0 = insertelement <2 x i16> undef, i16 %a, i32 0
  %ins.1 = insertelement <2 x i16> %ins.0, i16 5, i32 1
  %r = bitcast <2 x i16> %ins.1 to <2 x half>
  ret <2 x half> %r
}


; COMMON-LABEL: test_shufflevector(
; COMMON:	ld.param.u32 	[[R:%r[0-9]+]], [test_shufflevector_param_0];
; COMMON:	mov.b32 	{[[RS0:%rs[0-9]+]], [[RS1:%rs[0-9]+]]}, [[R]];
; COMMON:	mov.b32 	[[R1:%r[0-9]+]], {[[RS1]], [[RS0]]};
; COMMON:	st.param.b32 	[func_retval0+0], [[R1]];
; COMMON:	ret;
define <2 x i16> @test_shufflevector(<2 x i16> %a) #0 {
  %s = shufflevector <2 x i16> %a, <2 x i16> undef, <2 x i32> <i32 1, i32 0>
  ret <2 x i16> %s
}

; COMMON-LABEL: test_insertelement(
; COMMON:  ld.param.u16 	[[B:%rs[0-9]+]], [test_insertelement_param_1];
; COMMON:	ld.param.u32 	[[A:%r[0-9]+]], [test_insertelement_param_0];
; COMMON:	{ .reg .b16 tmp; mov.b32 {[[R0:%rs[0-9]+]], tmp}, [[A]]; }
; COMMON:	mov.b32 	[[R1:%r[0-9]+]], {[[R0]], [[B]]};
; COMMON:	st.param.b32 	[func_retval0+0], [[R1]];
; COMMON:	ret;
define <2 x i16> @test_insertelement(<2 x i16> %a, i16 %x) #0 {
  %i = insertelement <2 x i16> %a, i16 %x, i64 1
  ret <2 x i16> %i
}

; COMMON-LABEL: test_fptosi_2xhalf_to_2xi16(
; COMMON:      cvt.rzi.s16.f16
; COMMON:      cvt.rzi.s16.f16
; COMMON:      ret;
define <2 x i16> @test_fptosi_2xhalf_to_2xi16(<2 x half> %a) #0 {
  %r = fptosi <2 x half> %a to <2 x i16>
  ret <2 x i16> %r
}

; COMMON-LABEL: test_fptoui_2xhalf_to_2xi16(
; COMMON:      cvt.rzi.u16.f16
; COMMON:      cvt.rzi.u16.f16
; COMMON:      ret;
define <2 x i16> @test_fptoui_2xhalf_to_2xi16(<2 x half> %a) #0 {
  %r = fptoui <2 x half> %a to <2 x i16>
  ret <2 x i16> %r
}

attributes #0 = { nounwind }
