; RUN: llc -mtriple=amdgcn--amdpal -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SDAG -enable-var-scope %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SDAG -enable-var-scope %s
; RUN: llc -global-isel -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GISEL -enable-var-scope %s

; Make sure this interacts well with -amdgpu-fixed-function-abi
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -amdgpu-fixed-function-abi -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SDAG -enable-var-scope %s

declare float @extern_func(float) #0
declare float @extern_func_many_args(<64 x float>) #0

@funcptr = external hidden unnamed_addr addrspace(4) constant void()*, align 4

define amdgpu_gfx float @no_stack(float %arg0) #0 {
  %add = fadd float %arg0, 1.0
  ret float %add
}

define amdgpu_gfx float @simple_stack(float %arg0) #0 {
  %stack = alloca float, i32 4, align 4, addrspace(5)
  store volatile float 2.0, float addrspace(5)* %stack
  %val = load volatile float, float addrspace(5)* %stack
  %add = fadd float %arg0, %val
  ret float %add
}

define amdgpu_gfx float @multiple_stack(float %arg0) #0 {
  %stack = alloca float, i32 4, align 4, addrspace(5)
  store volatile float 2.0, float addrspace(5)* %stack
  %val = load volatile float, float addrspace(5)* %stack
  %add = fadd float %arg0, %val
  %stack2 = alloca float, i32 4, align 4, addrspace(5)
  store volatile float 2.0, float addrspace(5)* %stack2
  %val2 = load volatile float, float addrspace(5)* %stack2
  %add2 = fadd float %add, %val2
  ret float %add2
}

define amdgpu_gfx float @dynamic_stack(float %arg0) #0 {
bb0:
  %cmp = fcmp ogt float %arg0, 0.0
  br i1 %cmp, label %bb1, label %bb2

bb1:
  %stack = alloca float, i32 4, align 4, addrspace(5)
  store volatile float 2.0, float addrspace(5)* %stack
  %val = load volatile float, float addrspace(5)* %stack
  %add = fadd float %arg0, %val
  br label %bb2

bb2:
  %res = phi float [ 0.0, %bb0 ], [ %add, %bb1 ]
  ret float %res
}

define amdgpu_gfx float @dynamic_stack_loop(float %arg0) #0 {
bb0:
  br label %bb1

bb1:
  %ctr = phi i32 [ 0, %bb0 ], [ %newctr, %bb1 ]
  %stack = alloca float, i32 4, align 4, addrspace(5)
  store volatile float 2.0, float addrspace(5)* %stack
  %val = load volatile float, float addrspace(5)* %stack
  %add = fadd float %arg0, %val
  %cmp = icmp sgt i32 %ctr, 0
  %newctr = sub i32 %ctr, 1
  br i1 %cmp, label %bb1, label %bb2

bb2:
  ret float %add
}

define amdgpu_gfx float @no_stack_call(float %arg0) #0 {
  %res = call amdgpu_gfx float @simple_stack(float %arg0)
  ret float %res
}

define amdgpu_gfx float @simple_stack_call(float %arg0) #0 {
  %stack = alloca float, i32 4, align 4, addrspace(5)
  store volatile float 2.0, float addrspace(5)* %stack
  %val = load volatile float, float addrspace(5)* %stack
  %res = call amdgpu_gfx float @simple_stack(float %arg0)
  %add = fadd float %res, %val
  ret float %add
}

define amdgpu_gfx float @no_stack_extern_call(float %arg0) #0 {
  %res = call amdgpu_gfx float @extern_func(float %arg0)
  ret float %res
}

define amdgpu_gfx float @simple_stack_extern_call(float %arg0) #0 {
  %stack = alloca float, i32 4, align 4, addrspace(5)
  store volatile float 2.0, float addrspace(5)* %stack
  %val = load volatile float, float addrspace(5)* %stack
  %res = call amdgpu_gfx float @extern_func(float %arg0)
  %add = fadd float %res, %val
  ret float %add
}

define amdgpu_gfx float @no_stack_extern_call_many_args(<64 x float> %arg0) #0 {
  %res = call amdgpu_gfx float @extern_func_many_args(<64 x float> %arg0)
  ret float %res
}

define amdgpu_gfx float @no_stack_indirect_call(float %arg0) #0 {
  %fptr = load void()*, void()* addrspace(4)* @funcptr
  call amdgpu_gfx void %fptr()
  ret float %arg0
}

define amdgpu_gfx float @simple_stack_indirect_call(float %arg0) #0 {
  %stack = alloca float, i32 4, align 4, addrspace(5)
  store volatile float 2.0, float addrspace(5)* %stack
  %val = load volatile float, float addrspace(5)* %stack
  %fptr = load void()*, void()* addrspace(4)* @funcptr
  call amdgpu_gfx void %fptr()
  %add = fadd float %arg0, %val
  ret float %add
}

define amdgpu_gfx float @simple_stack_recurse(float %arg0) #0 {
  %stack = alloca float, i32 4, align 4, addrspace(5)
  store volatile float 2.0, float addrspace(5)* %stack
  %val = load volatile float, float addrspace(5)* %stack
  %res = call amdgpu_gfx float @simple_stack_recurse(float %arg0)
  %add = fadd float %res, %val
  ret float %add
}

@lds = internal addrspace(3) global [64 x float] undef

define amdgpu_gfx float @simple_lds(float %arg0) #0 {
  %lds_ptr = getelementptr [64 x float], [64 x float] addrspace(3)* @lds, i32 0, i32 0
  %val = load float, float addrspace(3)* %lds_ptr
  ret float %val
}

define amdgpu_gfx float @simple_lds_recurse(float %arg0) #0 {
  %lds_ptr = getelementptr [64 x float], [64 x float] addrspace(3)* @lds, i32 0, i32 0
  %val = load float, float addrspace(3)* %lds_ptr
  %res = call amdgpu_gfx float @simple_lds_recurse(float %val)
  ret float %res
}

attributes #0 = { nounwind }

; GCN: amdpal.pipelines:
; GCN-NEXT:  - .registers:
; SDAG-NEXT:      0x2e12 (COMPUTE_PGM_RSRC1): 0xaf03cf{{$}}
; SDAG-NEXT:      0x2e13 (COMPUTE_PGM_RSRC2): 0x8001{{$}}
; GISEL-NEXT:      0x2e12 (COMPUTE_PGM_RSRC1): 0xaf03cf{{$}}
; GISEL-NEXT:      0x2e13 (COMPUTE_PGM_RSRC2): 0x8001{{$}}
; GCN-NEXT:    .shader_functions:
; GCN-NEXT:      dynamic_stack:
; GCN-NEXT:        .stack_frame_size_in_bytes: 0x10{{$}}
; GCN-NEXT:      dynamic_stack_loop:
; GCN-NEXT:        .stack_frame_size_in_bytes: 0x10{{$}}
; GCN-NEXT:      multiple_stack:
; GCN-NEXT:        .stack_frame_size_in_bytes: 0x24{{$}}
; GCN-NEXT:      no_stack:
; GCN-NEXT:        .stack_frame_size_in_bytes: 0{{$}}
; GCN-NEXT:      no_stack_call:
; GCN-NEXT:        .stack_frame_size_in_bytes: 0{{$}}
; GCN-NEXT:      no_stack_extern_call:
; GCN-NEXT:        .stack_frame_size_in_bytes: 0x10{{$}}
; GCN-NEXT:      no_stack_extern_call_many_args:
; SDAG-NEXT:        .stack_frame_size_in_bytes: 0x90{{$}}
; GISEL-NEXT:        .stack_frame_size_in_bytes: 0x90{{$}}
; GCN-NEXT:      no_stack_indirect_call:
; GCN-NEXT:        .stack_frame_size_in_bytes: 0x10{{$}}
; GCN-NEXT:      simple_lds:
; GCN-NEXT:        .stack_frame_size_in_bytes: 0{{$}}
; GCN-NEXT:      simple_lds_recurse:
; GCN-NEXT:        .stack_frame_size_in_bytes: 0x10{{$}}
; GCN-NEXT:      simple_stack:
; GCN-NEXT:        .stack_frame_size_in_bytes: 0x14{{$}}
; GCN-NEXT:      simple_stack_call:
; GCN-NEXT:        .stack_frame_size_in_bytes: 0x20{{$}}
; GCN-NEXT:      simple_stack_extern_call:
; GCN-NEXT:        .stack_frame_size_in_bytes: 0x20{{$}}
; GCN-NEXT:      simple_stack_indirect_call:
; GCN-NEXT:        .stack_frame_size_in_bytes: 0x20{{$}}
; GCN-NEXT:      simple_stack_recurse:
; GCN-NEXT:        .stack_frame_size_in_bytes: 0x20{{$}}
; GCN-NEXT: ...
