; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -mattr=-flat-for-global -enable-ipra=0 -amdgpu-sroa=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,CIVI,MESA %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -enable-ipra=0 -amdgpu-sroa=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI,CIVI,MESA %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-flat-for-global -enable-ipra=0 -amdgpu-sroa=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,MESA %s
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
define hidden fastcc i32 @i32_fastcc_i32_byval_i32(i32 %arg0, i32 addrspace(5)* byval align 4 %arg1) #1 {
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
define fastcc i32 @sibling_call_i32_fastcc_i32_byval_i32_byval_parent(i32 %a, i32 addrspace(5)* byval %b.byval, i32 %c) #1 {
entry:
  %ret = tail call fastcc i32 @i32_fastcc_i32_byval_i32(i32 %a, i32 addrspace(5)* %b.byval)
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
  %ret = tail call fastcc i32 @i32_fastcc_i32_byval_i32(i32 %a, i32 addrspace(5)* inttoptr (i32 16 to i32 addrspace(5)*))
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

; GCN-DAG: buffer_store_dword v32, off, s[0:3], s32 offset:12 ; 4-byte Folded Spill
; GCN-DAG: buffer_store_dword v33, off, s[0:3], s32 offset:8 ; 4-byte Folded Spill

; GCN-DAG: buffer_load_dword [[LOAD_0:v[0-9]+]], off, s[0:3], s32{{$}}
; GCN-DAG: buffer_load_dword [[LOAD_1:v[0-9]+]], off, s[0:3], s32 offset:4

; GCN-NOT: s32

; GCN-DAG: buffer_store_dword [[LOAD_0]], off, s[0:3], s32{{$}}
; GCN-DAG: buffer_store_dword [[LOAD_1]], off, s[0:3], s32 offset:4

; GCN-DAG: buffer_load_dword v32, off, s[0:3], s32 offset:12 ; 4-byte Folded Reload
; GCN-DAG: buffer_load_dword v33, off, s[0:3], s32 offset:8 ; 4-byte Folded Reload

; GCN-NOT: s32
; GCN: s_setpc_b64
define fastcc i32 @sibling_call_i32_fastcc_i32_i32_a32i32(i32 %a, i32 %b, [32 x i32] %c) #1 {
entry:
  %ret = tail call fastcc i32 @i32_fastcc_i32_i32_a32i32(i32 %a, i32 %b, [32 x i32] %c)
  ret i32 %ret
}

; GCN-LABEL: {{^}}sibling_call_i32_fastcc_i32_i32_a32i32_stack_object:
; GCN-DAG: v_mov_b32_e32 [[NINE:v[0-9]+]], 9
; GCN: buffer_store_dword [[NINE]], off, s[0:3], s32 offset:40
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
; GCN-NEXT: buffer_store_dword v34, off, s[0:3], s32 offset:8 ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec
; GCN: s_mov_b32 s33, s32
; GCN-DAG: s_add_u32 s32, s32, 0x400

; GCN-DAG: buffer_store_dword v32, off, s[0:3], s33 offset:4 ; 4-byte Folded Spill
; GCN-DAG: buffer_store_dword v33, off, s[0:3], s33 ; 4-byte Folded Spill
; GCN-DAG: v_writelane_b32 v34, s34, 0
; GCN-DAG: v_writelane_b32 v34, s35, 1

; GCN-DAG: s_getpc_b64 s[4:5]
; GCN-DAG: s_add_u32 s4, s4, i32_fastcc_i32_i32@gotpcrel32@lo+4
; GCN-DAG: s_addc_u32 s5, s5, i32_fastcc_i32_i32@gotpcrel32@hi+4


; GCN: s_swappc_b64

; GCN-DAG: v_readlane_b32 s34, v34, 0
; GCN-DAG: v_readlane_b32 s35, v34, 1

; GCN: buffer_load_dword v33, off, s[0:3], s33 ; 4-byte Folded Reload
; GCN: buffer_load_dword v32, off, s[0:3], s33 offset:4 ; 4-byte Folded Reload

; GCN: s_getpc_b64 s[4:5]
; GCN-NEXT: s_add_u32 s4, s4, sibling_call_i32_fastcc_i32_i32@rel32@lo+4
; GCN-NEXT: s_addc_u32 s5, s5, sibling_call_i32_fastcc_i32_i32@rel32@hi+4

; GCN: s_sub_u32 s32, s32, 0x400
; GCN-NEXT: v_readlane_b32 s33,
; GCN-NEXT: s_or_saveexec_b64 s[6:7], -1
; GCN-NEXT: buffer_load_dword v34, off, s[0:3], s32 offset:8 ; 4-byte Folded Reload
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
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s32 offset:

; GCN-NOT: s33

; GCN: buffer_load_dword v{{[0-9]+}}, off, s[0:3], s32 offset:
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

attributes #0 = { nounwind }
attributes #1 = { nounwind noinline }
