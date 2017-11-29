; RUN: llc -mtriple=amdgcn-amd-amdhsa-amdgiz -mcpu=fiji -mattr=-flat-for-global -enable-ipra=0 -amdgpu-sroa=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,MESA %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa-amdgiz -mcpu=hawaii -enable-ipra=0 -amdgpu-sroa=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI,MESA %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa-amdgiz -mcpu=gfx900 -mattr=-flat-for-global -enable-ipra=0 -amdgpu-sroa=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,VI,MESA %s
target datalayout = "A5"

; GCN-LABEL: {{^}}i32_fastcc_i32_i32:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT: v_add_{{[_coiu]*}}32_e32 v0, vcc, v1, v0
; GCN-NEXT: s_setpc_b64
define fastcc i32 @i32_fastcc_i32_i32(i32 %arg0, i32 %arg1) #1 {
  %add0 = add i32 %arg0, %arg1
  ret i32 %add0
}

; GCN-LABEL: {{^}}i32_fastcc_i32_i32_stack_object:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN: v_add_{{[_coiu]*}}32_e32 v0, vcc, v1, v
; GCN: s_mov_b32 s5, s32
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s5 offset:24
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
define fastcc i32 @sibling_call_i32_fastcc_i32_i32(i32 %a, i32 %b, i32 %c) #1 {
entry:
  %ret = tail call fastcc i32 @i32_fastcc_i32_i32(i32 %a, i32 %b)
  ret i32 %ret
}

; GCN-LABEL: {{^}}sibling_call_i32_fastcc_i32_i32_stack_object:
; GCN: v_mov_b32_e32 [[NINE:v[0-9]+]], 9
; GCN: buffer_store_dword [[NINE]], off, s[0:3], s5 offset:24
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
; GCN: buffer_store_dword [[NINE]], off, s[0:3], s5 offset:24
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
; GCN-NEXT: s_mov_b32 s5, s32
; GCN-NEXT: buffer_load_dword v1, off, s[0:3], s5 offset:4
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: v_add_{{[_coiu]*}}32_e32 v0, vcc, v1, v0
; GCN-NEXT: s_setpc_b64 s[30:31]
define fastcc i32 @i32_fastcc_i32_byval_i32(i32 %arg0, i32 addrspace(5)* byval align 4 %arg1) #1 {
  %arg1.load = load i32, i32 addrspace(5)* %arg1, align 4
  %add0 = add i32 %arg0, %arg1.load
  ret i32 %add0
}

; Tail call disallowed with byval in parent.
; GCN-LABEL: {{^}}sibling_call_i32_fastcc_i32_byval_i32_byval_parent:
; GCN-NOT: v_writelane_b32 v{{[0-9]+}}, s32
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s32 offset:4
; GCN: s_swappc_b64
; GCN-NOT: v_readlane_b32 s32
; GCN: s_setpc_b64
define fastcc i32 @sibling_call_i32_fastcc_i32_byval_i32_byval_parent(i32 %a, i32 addrspace(5)* byval %b.byval, i32 %c) #1 {
entry:
  %ret = tail call fastcc i32 @i32_fastcc_i32_byval_i32(i32 %a, i32 addrspace(5)* %b.byval)
  ret i32 %ret
}

; Tail call disallowed with byval in parent, not callee.
; GCN-LABEL: {{^}}sibling_call_i32_fastcc_i32_byval_i32:
; GCN-NOT: v0
; GCN-NOT: s32
; GCN: buffer_load_dword v1, off, s[0:3], s4 offset:16
; GCN: s_mov_b32 s5, s32
; GCN: buffer_store_dword v1, off, s[0:3], s5 offset:4
; GCN-NEXT: s_setpc_b64
define fastcc i32 @sibling_call_i32_fastcc_i32_byval_i32(i32 %a, [16 x i32] %large) #1 {
entry:
  %ret = tail call fastcc i32 @i32_fastcc_i32_byval_i32(i32 %a, i32 addrspace(5)* inttoptr (i32 16 to i32 addrspace(5)*))
  ret i32 %ret
}

; GCN-LABEL: {{^}}i32_fastcc_i32_i32_a32i32:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-DAG: buffer_load_dword [[LOAD_0:v[0-9]+]], off, s[0:3], s5 offset:4
; GCN-DAG: buffer_load_dword [[LOAD_1:v[0-9]+]], off, s[0:3], s5 offset:8
; GCN-DAG: v_add_{{[_coiu]*}}32_e32 v0, vcc, v1, v0
; GCN: v_add_{{[_coiu]*}}32_e32 v0, vcc, [[LOAD_0]], v0
; GCN: v_add_{{[_coiu]*}}32_e32 v0, vcc, [[LOAD_1]], v0
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
; GCN: s_mov_b32 s5, s32

; GCN-DAG: buffer_store_dword v32, off, s[0:3], s5 offset:16 ; 4-byte Folded Spill
; GCN-DAG: buffer_store_dword v33, off, s[0:3], s5 offset:12 ; 4-byte Folded Spill

; GCN-DAG: buffer_load_dword [[LOAD_0:v[0-9]+]], off, s[0:3], s5 offset:4
; GCN-DAG: buffer_load_dword [[LOAD_1:v[0-9]+]], off, s[0:3], s5 offset:8

; GCN-NOT: s32

; GCN-DAG: buffer_store_dword [[LOAD_0]], off, s[0:3], s5 offset:4
; GCN-DAG: buffer_store_dword [[LOAD_1]], off, s[0:3], s5 offset:8

; GCN-DAG: buffer_load_dword v32, off, s[0:3], s5 offset:16 ; 4-byte Folded Reload
; GCN-DAG: buffer_load_dword v33, off, s[0:3], s5 offset:12 ; 4-byte Folded Reload

; GCN-NOT: s32
; GCN: s_setpc_b64
define fastcc i32 @sibling_call_i32_fastcc_i32_i32_a32i32(i32 %a, i32 %b, [32 x i32] %c) #1 {
entry:
  %ret = tail call fastcc i32 @i32_fastcc_i32_i32_a32i32(i32 %a, i32 %b, [32 x i32] %c)
  ret i32 %ret
}

; GCN-LABEL: {{^}}sibling_call_i32_fastcc_i32_i32_a32i32_stack_object:
; GCN-DAG: s_mov_b32 s5, s32
; GCN-NOT: s32
; GCN-DAG: v_mov_b32_e32 [[NINE:v[0-9]+]], 9
; GCN: buffer_store_dword [[NINE]], off, s[0:3], s5 offset:44

; GCN-NOT: s32
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
; GCN: s_mov_b32 s5, s32
; GCN: buffer_store_dword v34, off, s[0:3], s5 offset:12
; GCN: buffer_store_dword v32, off, s[0:3], s5 offset:8 ; 4-byte Folded Spill
; GCN: buffer_store_dword v33, off, s[0:3], s5 offset:4 ; 4-byte Folded Spill
; GCN-DAG: v_writelane_b32 v34, s33, 0
; GCN-DAG: v_writelane_b32 v34, s34, 1
; GCN-DAG: v_writelane_b32 v34, s35, 2
; GCN-DAG: s_add_u32 s32, s32, 0x400

; GCN: s_getpc_b64
; GCN: s_swappc_b64

; GCN: s_getpc_b64 s[6:7]
; GCN: s_add_u32 s6, s6, sibling_call_i32_fastcc_i32_i32@rel32@lo+4
; GCN: s_addc_u32 s7, s7, sibling_call_i32_fastcc_i32_i32@rel32@hi+4

; GCN-DAG: v_readlane_b32 s33, v34, 0
; GCN-DAG: v_readlane_b32 s34, v34, 1
; GCN-DAG: v_readlane_b32 s35, v34, 2

; GCN: buffer_load_dword v33, off, s[0:3], s5 offset:4
; GCN: buffer_load_dword v32, off, s[0:3], s5 offset:8
; GCN: buffer_load_dword v34, off, s[0:3], s5 offset:12
; GCN: s_sub_u32 s32, s32, 0x400
; GCN: s_setpc_b64 s[6:7]
define fastcc i32 @sibling_call_i32_fastcc_i32_i32_other_call(i32 %a, i32 %b, i32 %c) #1 {
entry:
  %other.call = tail call fastcc i32 @i32_fastcc_i32_i32(i32 %a, i32 %b)
  %ret = tail call fastcc i32 @sibling_call_i32_fastcc_i32_i32(i32 %a, i32 %b, i32 %other.call)
  ret i32 %ret
}

; Have stack object in caller and stack passed arguments. SP should be
; in same place at function exit.

; GCN-LABEL: {{^}}sibling_call_stack_objecti32_fastcc_i32_i32_a32i32:
; GCN: s_mov_b32 s5, s32
; GCN-NOT: s32
; GCN: s_setpc_b64 s[6:7]
define fastcc i32 @sibling_call_stack_objecti32_fastcc_i32_i32_a32i32(i32 %a, i32 %b, [32 x i32] %c) #1 {
entry:
  %alloca = alloca [16 x i32], align 4, addrspace(5)
  %gep = getelementptr inbounds [16 x i32], [16 x i32] addrspace(5)* %alloca, i32 0, i32 5
  store volatile i32 9, i32 addrspace(5)* %gep
  %ret = tail call fastcc i32 @i32_fastcc_i32_i32_a32i32(i32 %a, i32 %b, [32 x i32] %c)
  ret i32 %ret
}

; GCN-LABEL: {{^}}sibling_call_stack_objecti32_fastcc_i32_i32_a32i32_larger_arg_area:
; GCN: s_mov_b32 s5, s32
; GCN-NOT: s32
; GCN: s_setpc_b64 s[6:7]
define fastcc i32 @sibling_call_stack_objecti32_fastcc_i32_i32_a32i32_larger_arg_area(i32 %a, i32 %b, [36 x i32] %c) #1 {
entry:
  %alloca = alloca [16 x i32], align 4, addrspace(5)
  %gep = getelementptr inbounds [16 x i32], [16 x i32] addrspace(5)* %alloca, i32 0, i32 5
  store volatile i32 9, i32 addrspace(5)* %gep
  %ret = tail call fastcc i32 @i32_fastcc_i32_i32_a32i32(i32 %a, i32 %b, [32 x i32] zeroinitializer)
  ret i32 %ret
}

attributes #0 = { nounwind }
attributes #1 = { nounwind noinline }
