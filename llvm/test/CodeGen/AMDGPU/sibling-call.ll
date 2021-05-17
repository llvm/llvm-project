; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -mattr=-flat-for-global -enable-ipra=0 -amdgpu-sroa=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CIVI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -enable-ipra=0 -amdgpu-sroa=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CIVI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-flat-for-global -enable-ipra=0 -amdgpu-sroa=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s
target datalayout = "A5"

; FIXME: Why is this commuted only sometimes?
; GCN-LABEL: {{^}}i32_fastcc_i32_i32:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CIVI-NEXT: v_add_{{i|u}}32_e32 v0, vcc, v0, v1
; GFX9-NEXT: v_add_u32_e32 v0, v0, v1
; GCN-NEXT: s_setpc_b64
define fastcc i32 @i32_fastcc_i32_i32(i32 %arg0, i32 %arg1) #1 {
  %add0 = add i32 %arg0, %arg1
  ret i32 %add0
}

; GCN-LABEL: {{^}}i32_fastcc_i32_i32_stack_object:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT: v_mov_b32_e32 [[K:v[0-9]+]], 9
; CIVI-NEXT: v_add_{{i|u}}32_e32 v0, vcc, v0, v1
; GFX9-NEXT: v_add_u32_e32 v0, v0, v1
; GCN: buffer_store_dword [[K]], off, s[0:3], s32 offset:20
; GCN: s_waitcnt vmcnt(0)
; GCN: s_setpc_b64
; GCN: ; ScratchSize: 68
define fastcc i32 @i32_fastcc_i32_i32_stack_object(i32 %arg0, i32 %arg1) #1 {
  %alloca = alloca [16 x i32], align 4, addrspace(5)
  %gep = getelementptr inbounds [16 x i32], [16 x i32] addrspace(5)* %alloca, i32 0, i32 5
  store volatile i32 9, i32 addrspace(5)* %gep
  %add0 = add i32 %arg0, %arg1
  ret i32 %add0
}

; GCN-LABEL: {{^}}sibling_call_i32_fastcc_i32_i32:
define hidden fastcc i32 @sibling_call_i32_fastcc_i32_i32(i32 %a, i32 %b, i32 %c) #1 {
entry:
  %ret = tail call fastcc i32 @i32_fastcc_i32_i32(i32 %a, i32 %b)
  ret i32 %ret
}

; GCN-LABEL: {{^}}sibling_call_i32_fastcc_i32_i32_stack_object:
; GCN: v_mov_b32_e32 [[NINE:v[0-9]+]], 9
; GCN: buffer_store_dword [[NINE]], off, s[0:3], s32 offset:20
; GCN: s_setpc_b64
; GCN: ; ScratchSize: 68
define fastcc i32 @sibling_call_i32_fastcc_i32_i32_stack_object(i32 %a, i32 %b, i32 %c) #1 {
entry:
  %alloca = alloca [16 x i32], align 4, addrspace(5)
  %gep = getelementptr inbounds [16 x i32], [16 x i32] addrspace(5)* %alloca, i32 0, i32 5
  store volatile i32 9, i32 addrspace(5)* %gep
  %ret = tail call fastcc i32 @i32_fastcc_i32_i32(i32 %a, i32 %b)
  ret i32 %ret
}

; GCN-LABEL: {{^}}sibling_call_i32_fastcc_i32_i32_callee_stack_object:
; GCN: v_mov_b32_e32 [[NINE:v[0-9]+]], 9
; GCN: buffer_store_dword [[NINE]], off, s[0:3], s32 offset:20
; GCN: s_setpc_b64
; GCN: ; ScratchSize: 136
define fastcc i32 @sibling_call_i32_fastcc_i32_i32_callee_stack_object(i32 %a, i32 %b, i32 %c) #1 {
entry:
  %alloca = alloca [16 x i32], align 4, addrspace(5)
  %gep = getelementptr inbounds [16 x i32], [16 x i32] addrspace(5)* %alloca, i32 0, i32 5
  store volatile i32 9, i32 addrspace(5)* %gep
  %ret = tail call fastcc i32 @i32_fastcc_i32_i32_stack_object(i32 %a, i32 %b)
  ret i32 %ret
}

; GCN-LABEL: {{^}}sibling_call_i32_fastcc_i32_i32_unused_result:
define fastcc void @sibling_call_i32_fastcc_i32_i32_unused_result(i32 %a, i32 %b, i32 %c) #1 {
entry:
  %ret = tail call fastcc i32 @i32_fastcc_i32_i32(i32 %a, i32 %b)
  ret void
}

; It doesn't make sense to do a tail from a kernel
; GCN-LABEL: {{^}}kernel_call_i32_fastcc_i32_i32_unused_result:
;define amdgpu_kernel void @kernel_call_i32_fastcc_i32_i32_unused_result(i32 %a, i32 %b, i32 %c) #1 {
define amdgpu_kernel void @kernel_call_i32_fastcc_i32_i32_unused_result(i32 %a, i32 %b, i32 %c) #1 {
entry:
  %ret = tail call fastcc i32 @i32_fastcc_i32_i32(i32 %a, i32 %b)
  ret void
}

; GCN-LABEL: {{^}}i32_fastcc_i32_byval_i32:
; GCN: s_waitcnt
; GCN-NEXT: buffer_load_dword v1, off, s[0:3], s32{{$}}
; GCN-NEXT: s_waitcnt vmcnt(0)

; CIVI-NEXT: v_add_{{i|u}}32_e32 v0, vcc, v0, v1
; GFX9-NEXT: v_add_u32_e32 v0, v0, v1

; GCN-NEXT: s_setpc_b64 s[30:31]
define hidden fastcc i32 @i32_fastcc_i32_byval_i32(i32 %arg0, i32 addrspace(5)* byval(i32) align 4 %arg1) #1 {
  %arg1.load = load i32, i32 addrspace(5)* %arg1, align 4
  %add0 = add i32 %arg0, %arg1.load
  ret i32 %add0
}

; Tail call disallowed with byval in parent.
; GCN-LABEL: {{^}}sibling_call_i32_fastcc_i32_byval_i32_byval_parent:
; GCN-NOT: v_writelane_b32 v{{[0-9]+}}, s32
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s32{{$}}
; GCN: s_swappc_b64
; GCN-NOT: v_readlane_b32 s32
; GCN: s_setpc_b64
define fastcc i32 @sibling_call_i32_fastcc_i32_byval_i32_byval_parent(i32 %a, i32 addrspace(5)* byval(i32) %b.byval, i32 %c) #1 {
entry:
  %ret = tail call fastcc i32 @i32_fastcc_i32_byval_i32(i32 %a, i32 addrspace(5)* byval(i32) %b.byval)
  ret i32 %ret
}

; Tail call disallowed with byval in parent, not callee. The stack
; usage of incoming arguments must be <= the outgoing stack
; arguments.

; GCN-LABEL: {{^}}sibling_call_i32_fastcc_i32_byval_i32:
; GCN-NOT: v0
; GCN-NOT: s32
; GCN: buffer_load_dword v1, off, s[0:3], 0 offset:16
; GCN: buffer_store_dword v1, off, s[0:3], s32{{$}}
; GCN-NEXT: s_setpc_b64
define fastcc i32 @sibling_call_i32_fastcc_i32_byval_i32(i32 %a, [32 x i32] %large) #1 {
entry:
  %ret = tail call fastcc i32 @i32_fastcc_i32_byval_i32(i32 %a, i32 addrspace(5)* byval(i32) inttoptr (i32 16 to i32 addrspace(5)*))
  ret i32 %ret
}

; GCN-LABEL: {{^}}i32_fastcc_i32_i32_a32i32:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-DAG: buffer_load_dword [[LOAD_0:v[0-9]+]], off, s[0:3], s32{{$}}
; GCN-DAG: buffer_load_dword [[LOAD_1:v[0-9]+]], off, s[0:3], s32 offset:4

; CIVI-NEXT: v_add_{{i|u}}32_e32 v0, vcc, v0, v1
; CIVI: v_add_{{i|u}}32_e32 v0, vcc, v0, [[LOAD_0]]
; CIVI: v_add_{{i|u}}32_e32 v0, vcc, v0, [[LOAD_1]]


; GFX9-NEXT: v_add_u32_e32 v0, v0, v1
; GFX9: v_add3_u32 v0, v0, v3, v2

; GCN-NEXT: s_setpc_b64
define fastcc i32 @i32_fastcc_i32_i32_a32i32(i32 %arg0, i32 %arg1, [32 x i32] %large) #1 {
  %val_firststack = extractvalue [32 x i32] %large, 30
  %val_laststack = extractvalue [32 x i32] %large, 31
  %add0 = add i32 %arg0, %arg1
  %add1 = add i32 %add0, %val_firststack
  %add2 = add i32 %add1, %val_laststack
  ret i32 %add2
}

; FIXME: Why load and store same location for stack args?
; GCN-LABEL: {{^}}sibling_call_i32_fastcc_i32_i32_a32i32:

; GCN-DAG: buffer_load_dword [[LOAD_0:v[0-9]+]], off, s[0:3], s32{{$}}
; GCN-DAG: buffer_load_dword [[LOAD_1:v[0-9]+]], off, s[0:3], s32 offset:4

; GCN-NOT: s32

; GCN-DAG: buffer_store_dword [[LOAD_0]], off, s[0:3], s32{{$}}
; GCN-DAG: buffer_store_dword [[LOAD_1]], off, s[0:3], s32 offset:4

; GCN-NOT: s32
; GCN: s_setpc_b64
define fastcc i32 @sibling_call_i32_fastcc_i32_i32_a32i32(i32 %a, i32 %b, [32 x i32] %c) #1 {
entry:
  %ret = tail call fastcc i32 @i32_fastcc_i32_i32_a32i32(i32 %a, i32 %b, [32 x i32] %c)
  ret i32 %ret
}

; GCN-LABEL: {{^}}sibling_call_i32_fastcc_i32_i32_a32i32_stack_object:
; GCN-DAG: v_mov_b32_e32 [[NINE:v[0-9]+]], 9
; GCN: buffer_store_dword [[NINE]], off, s[0:3], s32 offset:28
; GCN: s_setpc_b64
define fastcc i32 @sibling_call_i32_fastcc_i32_i32_a32i32_stack_object(i32 %a, i32 %b, [32 x i32] %c) #1 {
entry:
  %alloca = alloca [16 x i32], align 4, addrspace(5)
  %gep = getelementptr inbounds [16 x i32], [16 x i32] addrspace(5)* %alloca, i32 0, i32 5
  store volatile i32 9, i32 addrspace(5)* %gep
  %ret = tail call fastcc i32 @i32_fastcc_i32_i32_a32i32(i32 %a, i32 %b, [32 x i32] %c)
  ret i32 %ret
}

; If the callee requires more stack argument space than the caller,
; don't do a tail call.
; TODO: Do we really need this restriction?

; GCN-LABEL: {{^}}no_sibling_call_callee_more_stack_space:
; GCN: s_swappc_b64
; GCN: s_setpc_b64
define fastcc i32 @no_sibling_call_callee_more_stack_space(i32 %a, i32 %b) #1 {
entry:
  %ret = tail call fastcc i32 @i32_fastcc_i32_i32_a32i32(i32 %a, i32 %b, [32 x i32] zeroinitializer)
  ret i32 %ret
}

; Have another non-tail in the function
; GCN-LABEL: {{^}}sibling_call_i32_fastcc_i32_i32_other_call:
; GCN: s_or_saveexec_b64 s{{\[[0-9]+:[0-9]+\]}}, -1
; GCN-NEXT: buffer_store_dword v42, off, s[0:3], s32 offset:8 ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec
; GCN: s_mov_b32 s33, s32
; GCN-DAG: s_add_u32 s32, s32, 0x400

; GCN-DAG: buffer_store_dword v40, off, s[0:3], s33 offset:4 ; 4-byte Folded Spill
; GCN-DAG: buffer_store_dword v41, off, s[0:3], s33 ; 4-byte Folded Spill
; GCN-DAG: v_writelane_b32 v42, s34, 0
; GCN-DAG: v_writelane_b32 v42, s35, 1

; GCN-DAG: s_getpc_b64 s[4:5]
; GCN-DAG: s_add_u32 s4, s4, i32_fastcc_i32_i32@gotpcrel32@lo+4
; GCN-DAG: s_addc_u32 s5, s5, i32_fastcc_i32_i32@gotpcrel32@hi+12


; GCN: s_swappc_b64

; GCN-DAG: buffer_load_dword v41, off, s[0:3], s33 ; 4-byte Folded Reload
; GCN-DAG: buffer_load_dword v40, off, s[0:3], s33 offset:4 ; 4-byte Folded Reload

; GCN: s_getpc_b64 s[4:5]
; GCN-NEXT: s_add_u32 s4, s4, sibling_call_i32_fastcc_i32_i32@rel32@lo+4
; GCN-NEXT: s_addc_u32 s5, s5, sibling_call_i32_fastcc_i32_i32@rel32@hi+12

; GCN-DAG: v_readlane_b32 s34, v42, 0
; GCN-DAG: v_readlane_b32 s35, v42, 1

; GCN: s_sub_u32 s32, s32, 0x400
; GCN-NEXT: v_readlane_b32 s33,
; GCN-NEXT: s_or_saveexec_b64 s[6:7], -1
; GCN-NEXT: buffer_load_dword v42, off, s[0:3], s32 offset:8 ; 4-byte Folded Reload
; GCN-NEXT: s_mov_b64 exec, s[6:7]
; GCN-NEXT: s_setpc_b64 s[4:5]
define fastcc i32 @sibling_call_i32_fastcc_i32_i32_other_call(i32 %a, i32 %b, i32 %c) #1 {
entry:
  %other.call = tail call fastcc i32 @i32_fastcc_i32_i32(i32 %a, i32 %b)
  %ret = tail call fastcc i32 @sibling_call_i32_fastcc_i32_i32(i32 %a, i32 %b, i32 %other.call)
  ret i32 %ret
}

; Have stack object in caller and stack passed arguments. SP should be
; in same place at function exit.

; GCN-LABEL: {{^}}sibling_call_stack_objecti32_fastcc_i32_i32_a32i32:
; GCN-NOT: s33
; GCN: buffer_load_dword v{{[0-9]+}}, off, s[0:3], s32 offset:

; GCN-NOT: s33

; GCN: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s32 offset:
; GCN: s_setpc_b64 s[4:5]
define fastcc i32 @sibling_call_stack_objecti32_fastcc_i32_i32_a32i32(i32 %a, i32 %b, [32 x i32] %c) #1 {
entry:
  %alloca = alloca [16 x i32], align 4, addrspace(5)
  %gep = getelementptr inbounds [16 x i32], [16 x i32] addrspace(5)* %alloca, i32 0, i32 5
  store volatile i32 9, i32 addrspace(5)* %gep
  %ret = tail call fastcc i32 @i32_fastcc_i32_i32_a32i32(i32 %a, i32 %b, [32 x i32] %c)
  ret i32 %ret
}

; GCN-LABEL: {{^}}sibling_call_stack_objecti32_fastcc_i32_i32_a32i32_larger_arg_area:
; GCN-NOT: s33
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s32 offset:44

; GCN-NOT: s33
; GCN: s_setpc_b64 s[4:5]
define fastcc i32 @sibling_call_stack_objecti32_fastcc_i32_i32_a32i32_larger_arg_area(i32 %a, i32 %b, [36 x i32] %c) #1 {
entry:
  %alloca = alloca [16 x i32], align 4, addrspace(5)
  %gep = getelementptr inbounds [16 x i32], [16 x i32] addrspace(5)* %alloca, i32 0, i32 5
  store volatile i32 9, i32 addrspace(5)* %gep
  %ret = tail call fastcc i32 @i32_fastcc_i32_i32_a32i32(i32 %a, i32 %b, [32 x i32] zeroinitializer)
  ret i32 %ret
}

@func_ptr_gv = external unnamed_addr addrspace(4) constant i32(i32, i32)*, align 4

; Do support tail calls with a uniform, but unknown, callee.
; GCN-LABEL: {{^}}indirect_uniform_sibling_call_i32_fastcc_i32_i32:
; GCN: s_load_dwordx2 [[GV_ADDR:s\[[0-9]+:[0-9]+\]]]
; GCN: s_load_dwordx2 [[FUNC_PTR:s\[[0-9]+:[0-9]+\]]], [[GV_ADDR]]
; GCN: s_setpc_b64 [[FUNC_PTR]]
define hidden fastcc i32 @indirect_uniform_sibling_call_i32_fastcc_i32_i32(i32 %a, i32 %b, i32 %c) #1 {
entry:
  %func.ptr.load = load i32(i32, i32)*, i32(i32, i32)* addrspace(4)* @func_ptr_gv
  %ret = tail call fastcc i32 %func.ptr.load(i32 %a, i32 %b)
  ret i32 %ret
}

; We can't support a tail call to a divergent target. Use a waterfall
; loop around a regular call
; GCN-LABEL: {{^}}indirect_divergent_sibling_call_i32_fastcc_i32_i32:
; GCN: v_readfirstlane_b32
; GCN: v_readfirstlane_b32
; GCN: s_and_saveexec_b64
; GCN: s_swappc_b64
; GCN: s_cbranch_execnz
; GCN: s_setpc_b64
define hidden fastcc i32 @indirect_divergent_sibling_call_i32_fastcc_i32_i32(i32(i32, i32)* %func.ptr, i32 %a, i32 %b, i32 %c) #1 {
entry:
  %add = add i32 %b, %c
  %ret = tail call fastcc i32 %func.ptr(i32 %a, i32 %add)
  ret i32 %ret
}

declare hidden void @void_fastcc_multi_byval(i32 %a, [3 x i32] addrspace(5)* byval([3 x i32]) align 16, [2 x i64] addrspace(5)* byval([2 x i64]))

; GCN-LABEL: {{^}}sibling_call_fastcc_multi_byval:
; GCN-DAG: s_getpc_b64 [[TARGET_ADDR:s\[[0-9]+:[0-9]+\]]]
; GCN-DAG: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0
; GCN-DAG: v_mov_b32_e32 [[NINE:v[0-9]+]], 9

; GCN-DAG: buffer_store_dword [[NINE]], off, s[0:3], s32 offset:144
; GCN-DAG: buffer_store_dword [[NINE]], off, s[0:3], s32 offset:148
; GCN-DAG: buffer_store_dword [[NINE]], off, s[0:3], s32 offset:152

; GCN-DAG: buffer_store_dword [[NINE]], off, s[0:3], s32{{$}}
; GCN-DAG: buffer_store_dword [[NINE]], off, s[0:3], s32 offset:4{{$}}
; GCN-DAG: buffer_store_dword [[NINE]], off, s[0:3], s32 offset:8{{$}}

; GCN-DAG: buffer_store_dword [[ZERO]], off, s[0:3], s32 offset:160
; GCN-DAG: buffer_store_dword [[ZERO]], off, s[0:3], s32 offset:164
; GCN-DAG: buffer_store_dword [[ZERO]], off, s[0:3], s32 offset:168
; GCN-DAG: buffer_store_dword [[ZERO]], off, s[0:3], s32 offset:172
; GCN-DAG: buffer_store_dword [[ZERO]], off, s[0:3], s32 offset:16{{$}}
; GCN-DAG: buffer_store_dword [[ZERO]], off, s[0:3], s32 offset:20{{$}}
; GCN-DAG: buffer_store_dword [[ZERO]], off, s[0:3], s32 offset:24{{$}}
; GCN-DAG: buffer_store_dword [[ZERO]], off, s[0:3], s32 offset:28{{$}}

; GCN: s_setpc_b64 [[TARGET_ADDR]]
define fastcc void @sibling_call_fastcc_multi_byval(i32 %a, [64 x i32]) #1 {
entry:
  %alloca0 = alloca [3 x i32], align 16, addrspace(5)
  %alloca1 = alloca [2 x i64], align 8, addrspace(5)
  store [3 x i32] [i32 9, i32 9, i32 9], [3 x i32] addrspace(5)* %alloca0
  store [2 x i64] zeroinitializer, [2 x i64] addrspace(5)* %alloca1
  tail call fastcc void @void_fastcc_multi_byval(i32 %a, [3 x i32] addrspace(5)* byval([3 x i32]) %alloca0, [2 x i64] addrspace(5)* byval([2 x i64]) %alloca1)
  ret void
}

declare hidden void @void_fastcc_byval_and_stack_passed([3 x i32] addrspace(5)* byval([3 x i32]) align 16, [32 x i32], i32)

; Callee has a byval and non-byval stack passed argument
; GCN-LABEL: {{^}}sibling_call_byval_and_stack_passed:
; GCN: v_mov_b32_e32 [[NINE:v[0-9]+]], 9

; GCN-DAG: buffer_store_dword [[NINE]], off, s[0:3], s32 offset:144
; GCN-DAG: buffer_store_dword [[NINE]], off, s[0:3], s32 offset:148
; GCN-DAG: buffer_store_dword [[NINE]], off, s[0:3], s32 offset:152
; GCN-DAG: buffer_store_dword [[NINE]], off, s[0:3], s32{{$}}
; GCN-DAG: buffer_store_dword [[NINE]], off, s[0:3], s32 offset:4{{$}}
; GCN-DAG: buffer_store_dword [[NINE]], off, s[0:3], s32 offset:8{{$}}
; GCN-DAG: buffer_store_dword v0, off, s[0:3], s32 offset:12

; GCN: v_mov_b32_e32 v0, 0
; GCN: v_mov_b32_e32 v30, 0

; GCN: s_getpc_b64 [[TARGET_ADDR:s\[[0-9]+:[0-9]+\]]]
; GCN-NEXT: s_add_u32
; GCN-NEXT: s_addc_u32
; GCN-NEXT: s_setpc_b64 [[TARGET_ADDR]]
define fastcc void @sibling_call_byval_and_stack_passed(i32 %stack.out.arg, [64 x i32]) #1 {
entry:
  %alloca = alloca [3 x i32], align 16, addrspace(5)
  store [3 x i32] [i32 9, i32 9, i32 9], [3 x i32] addrspace(5)* %alloca
  tail call fastcc void @void_fastcc_byval_and_stack_passed([3 x i32] addrspace(5)* byval([3 x i32]) %alloca, [32 x i32] zeroinitializer, i32 %stack.out.arg)
  ret void
}

declare hidden fastcc i64 @i64_fastcc_i64(i64 %arg0)

; GCN-LABEL: {{^}}sibling_call_i64_fastcc_i64:
; GCN: s_waitcnt
; GCN-NEXT: s_getpc_b64
; GCN-NEXT: s_add_u32
; GCN-NEXT: s_addc_u32
; GCN-NEXT: s_setpc_b64
define hidden fastcc i64 @sibling_call_i64_fastcc_i64(i64 %a) #1 {
entry:
  %ret = tail call fastcc i64 @i64_fastcc_i64(i64 %a)
  ret i64 %ret
}

declare hidden fastcc i8 addrspace(1)* @p1i8_fastcc_p1i8(i8 addrspace(1)* %arg0)

; GCN-LABEL: {{^}}sibling_call_p1i8_fastcc_p1i8:
; GCN: s_waitcnt
; GCN-NEXT: s_getpc_b64
; GCN-NEXT: s_add_u32
; GCN-NEXT: s_addc_u32
; GCN-NEXT: s_setpc_b64
define hidden fastcc i8 addrspace(1)* @sibling_call_p1i8_fastcc_p1i8(i8 addrspace(1)* %a) #1 {
entry:
  %ret = tail call fastcc i8 addrspace(1)* @p1i8_fastcc_p1i8(i8 addrspace(1)* %a)
  ret i8 addrspace(1)* %ret
}

declare hidden fastcc i16 @i16_fastcc_i16(i16 %arg0)

; GCN-LABEL: {{^}}sibling_call_i16_fastcc_i16:
; GCN: s_waitcnt
; GCN-NEXT: s_getpc_b64
; GCN-NEXT: s_add_u32
; GCN-NEXT: s_addc_u32
; GCN-NEXT: s_setpc_b64
define hidden fastcc i16 @sibling_call_i16_fastcc_i16(i16 %a) #1 {
entry:
  %ret = tail call fastcc i16 @i16_fastcc_i16(i16 %a)
  ret i16 %ret
}

declare hidden fastcc half @f16_fastcc_f16(half %arg0)

; GCN-LABEL: {{^}}sibling_call_f16_fastcc_f16:
; GCN: s_waitcnt
; GCN-NEXT: s_getpc_b64
; GCN-NEXT: s_add_u32
; GCN-NEXT: s_addc_u32
; GCN-NEXT: s_setpc_b64
define hidden fastcc half @sibling_call_f16_fastcc_f16(half %a) #1 {
entry:
  %ret = tail call fastcc half @f16_fastcc_f16(half %a)
  ret half %ret
}

declare hidden fastcc <3 x i16> @v3i16_fastcc_v3i16(<3 x i16> %arg0)

; GCN-LABEL: {{^}}sibling_call_v3i16_fastcc_v3i16:
; GCN: s_waitcnt
; GCN-NEXT: s_getpc_b64
; GCN-NEXT: s_add_u32
; GCN-NEXT: s_addc_u32
; GCN-NEXT: s_setpc_b64
define hidden fastcc <3 x i16> @sibling_call_v3i16_fastcc_v3i16(<3 x i16> %a) #1 {
entry:
  %ret = tail call fastcc <3 x i16> @v3i16_fastcc_v3i16(<3 x i16> %a)
  ret <3 x i16> %ret
}

declare hidden fastcc <4 x i16> @v4i16_fastcc_v4i16(<4 x i16> %arg0)

; GCN-LABEL: {{^}}sibling_call_v4i16_fastcc_v4i16:
; GCN: s_waitcnt
; GCN-NEXT: s_getpc_b64
; GCN-NEXT: s_add_u32
; GCN-NEXT: s_addc_u32
; GCN-NEXT: s_setpc_b64
define hidden fastcc <4 x i16> @sibling_call_v4i16_fastcc_v4i16(<4 x i16> %a) #1 {
entry:
  %ret = tail call fastcc <4 x i16> @v4i16_fastcc_v4i16(<4 x i16> %a)
  ret <4 x i16> %ret
}

declare hidden fastcc <2 x i64> @v2i64_fastcc_v2i64(<2 x i64> %arg0)

; GCN-LABEL: {{^}}sibling_call_v2i64_fastcc_v2i64:
; GCN: s_waitcnt
; GCN-NEXT: s_getpc_b64
; GCN-NEXT: s_add_u32
; GCN-NEXT: s_addc_u32
; GCN-NEXT: s_setpc_b64
define hidden fastcc <2 x i64> @sibling_call_v2i64_fastcc_v2i64(<2 x i64> %a) #1 {
entry:
  %ret = tail call fastcc <2 x i64> @v2i64_fastcc_v2i64(<2 x i64> %a)
  ret <2 x i64> %ret
}

attributes #0 = { nounwind }
attributes #1 = { nounwind noinline }
