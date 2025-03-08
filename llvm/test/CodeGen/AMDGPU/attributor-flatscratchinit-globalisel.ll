; Test the generation of the attribute amdgpu-no-flat-scratch-init
; RUN: opt -S -O2 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -global-isel -stop-after=irtranslator | FileCheck -check-prefixes=GFX10 %s

;; tests of addrspacecast

define void @without_global_to_flat_addrspacecast(ptr addrspace(1) %ptr) {
  store volatile i32 0, ptr addrspace(1) %ptr
  ret void
}

define amdgpu_kernel void @without_global_to_flat_addrspacecast_cc_kernel(ptr addrspace(1) %ptr) {
  store volatile i32 0, ptr addrspace(1) %ptr
  ret void
}

define void @with_global_to_flat_addrspacecast(ptr addrspace(1) %ptr) {
  %stof = addrspacecast ptr addrspace(1) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define amdgpu_kernel void @with_global_to_flat_addrspacecast_cc_kernel(ptr addrspace(1) %ptr) {
  %stof = addrspacecast ptr addrspace(1) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define void @without_region_to_flat_addrspacecast(ptr addrspace(2) %ptr) {
  store volatile i32 0, ptr addrspace(2) %ptr
  ret void
}

define amdgpu_kernel void @without_region_to_flat_addrspacecast_cc_kernel(ptr addrspace(2) %ptr) {
  store volatile i32 0, ptr addrspace(2) %ptr
  ret void
}

define void @with_region_to_flat_addrspacecast(ptr addrspace(2) %ptr) {
  %stof = addrspacecast ptr addrspace(2) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define amdgpu_kernel void @with_region_to_flat_addrspacecast_cc_kernel(ptr addrspace(2) %ptr) {
  %stof = addrspacecast ptr addrspace(2) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define void @without_group_to_flat_addrspacecast(ptr addrspace(3) %ptr) {
  store volatile i32 0, ptr addrspace(3) %ptr
  ret void
}

define amdgpu_kernel void @without_group_to_flat_addrspacecast_cc_kernel(ptr addrspace(3) %ptr) {
  store volatile i32 0, ptr addrspace(3) %ptr
  ret void
}

define void @with_group_to_flat_addrspacecast(ptr addrspace(3) %ptr) {
  %stof = addrspacecast ptr addrspace(3) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define amdgpu_kernel void @with_group_to_flat_addrspacecast_cc_kernel(ptr addrspace(3) %ptr) {
  %stof = addrspacecast ptr addrspace(3) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define void @without_constant_to_flat_addrspacecast(ptr addrspace(4) %ptr) {
  store volatile i32 0, ptr addrspace(4) %ptr
  ret void
}

define amdgpu_kernel void @without_constant_to_flat_addrspacecast_cc_kernel(ptr addrspace(4) %ptr) {
  store volatile i32 0, ptr addrspace(4) %ptr
  ret void
}

define void @with_constant_to_flat_addrspacecast(ptr addrspace(4) %ptr) {
  %stof = addrspacecast ptr addrspace(4) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define amdgpu_kernel void @with_constant_to_flat_addrspacecast_cc_kernel(ptr addrspace(4) %ptr) {
  %stof = addrspacecast ptr addrspace(4) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
  store volatile i32 0, ptr addrspace(5) %ptr
  ret void
}

define amdgpu_kernel void @without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
  store volatile i32 0, ptr addrspace(5) %ptr
  ret void
}

define void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
  %stof = addrspacecast ptr addrspace(5) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define amdgpu_kernel void @with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
  %stof = addrspacecast ptr addrspace(5) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define void @call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_both_with_and_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @call_call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
  call void @call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
  call void @call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @call_call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
  call void @call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
  call void @call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @call_call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
  call void @call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_call_both_with_and_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
  call void @call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @with_cast_call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
  %stof = addrspacecast ptr addrspace(5) %ptr to ptr
  store volatile i32 0, ptr %stof
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @with_cast_call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
  %stof = addrspacecast ptr addrspace(5) %ptr to ptr
  store volatile i32 0, ptr %stof
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @with_cast_call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
  %stof = addrspacecast ptr addrspace(5) %ptr to ptr
  store volatile i32 0, ptr %stof
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @with_cast_call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
  %stof = addrspacecast ptr addrspace(5) %ptr to ptr
  store volatile i32 0, ptr %stof
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

;; tests of indirect call, intrinsics

@gv.fptr0 = external hidden unnamed_addr addrspace(4) constant ptr, align 4

define void @with_indirect_call() {
  %fptr = load ptr, ptr addrspace(4) @gv.fptr0
  call void %fptr()
  ret void
}

define amdgpu_kernel void @with_indirect_call_cc_kernel() {
  %fptr = load ptr, ptr addrspace(4) @gv.fptr0
  call void %fptr()
  ret void
}

define void @call_with_indirect_call() {
  call void @with_indirect_call()
  ret void
}

define amdgpu_kernel void @call_with_indirect_call_cc_kernel() {
  call void @with_indirect_call()
  ret void
}

declare i32 @llvm.amdgcn.workgroup.id.x()

define void @use_intrinsic_workitem_id_x() {
  %val = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %val, ptr addrspace(1) null
  ret void
}

define amdgpu_kernel void @use_intrinsic_workitem_id_x_cc_kernel() {
  %val = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %val, ptr addrspace(1) null
  ret void
}

define void @call_use_intrinsic_workitem_id_x() {
  call void @use_intrinsic_workitem_id_x()
  ret void
}

define amdgpu_kernel void @call_use_intrinsic_workitem_id_x_cc_kernel() {
  call void @use_intrinsic_workitem_id_x()
  ret void
}

; GFX10: name:            without_global_to_flat_addrspacecast
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            without_global_to_flat_addrspacecast_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr6' }
;
; GFX10: name:            with_global_to_flat_addrspacecast
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            with_global_to_flat_addrspacecast_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr6' }
; GFX10-NEXT:    privateSegmentWaveByteOffset: { reg: '$sgpr7' }
;
; GFX10: name:            without_region_to_flat_addrspacecast
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            without_region_to_flat_addrspacecast_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr6' }
;
; GFX10: name:            with_region_to_flat_addrspacecast
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            with_region_to_flat_addrspacecast_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr6' }
;
; GFX10: name:            without_group_to_flat_addrspacecast
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            without_group_to_flat_addrspacecast_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr6' }
;
; GFX10: name:            with_group_to_flat_addrspacecast
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            with_group_to_flat_addrspacecast_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr6' }
;
; GFX10: name:            without_constant_to_flat_addrspacecast
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            without_constant_to_flat_addrspacecast_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr6' }
;
; GFX10: name:            with_constant_to_flat_addrspacecast
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            with_constant_to_flat_addrspacecast_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr6' }
;
; GFX10: name:            without_private_to_flat_addrspacecast
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            without_private_to_flat_addrspacecast_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr6' }
;
; GFX10: name:            with_private_to_flat_addrspacecast
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            with_private_to_flat_addrspacecast_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr6' }
;
; GFX10: name:            call_without_private_to_flat_addrspacecast
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            call_without_private_to_flat_addrspacecast_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr6' }
;
; GFX10: name:            call_with_private_to_flat_addrspacecast
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            call_with_private_to_flat_addrspacecast_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr6' }
;
; GFX10: name:            call_both_with_and_without_private_to_flat_addrspacecast
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            call_both_with_and_without_private_to_flat_addrspacecast_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr6' }
;
; GFX10: name:            call_call_without_private_to_flat_addrspacecast
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            call_call_without_private_to_flat_addrspacecast_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr6' }
;
; GFX10: name:            call_call_with_private_to_flat_addrspacecast
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            call_call_with_private_to_flat_addrspacecast_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr6' }
;
; GFX10: name:            call_call_both_with_and_without_private_to_flat_addrspacecast
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            call_call_both_with_and_without_private_to_flat_addrspacecast_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr6' }
;
; GFX10: name:            with_cast_call_without_private_to_flat_addrspacecast
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            with_cast_call_without_private_to_flat_addrspacecast_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr6' }
;
; GFX10: name:            with_cast_call_with_private_to_flat_addrspacecast
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            with_cast_call_with_private_to_flat_addrspacecast_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr6' }
;
; GFX10: name:            with_indirect_call
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            with_indirect_call_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr8_sgpr9' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    flatScratchInit: { reg: '$sgpr12_sgpr13' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr14' }
;
; GFX10: name:            call_with_indirect_call
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            call_with_indirect_call_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    kernargSegmentPtr: { reg: '$sgpr8_sgpr9' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    flatScratchInit: { reg: '$sgpr12_sgpr13' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr14' }

;
; GFX10: name:            use_intrinsic_workitem_id_x
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            use_intrinsic_workitem_id_x_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr4' }
;
; GFX10: name:            call_use_intrinsic_workitem_id_x
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; GFX10-NEXT:    queuePtr:        { reg: '$sgpr6_sgpr7' }
; GFX10-NEXT:    dispatchID:      { reg: '$sgpr10_sgpr11' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr12' }
;
; GFX10: name:            call_use_intrinsic_workitem_id_x_cc_kernel
; GFX10:       argumentInfo:
; GFX10-NEXT:    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; GFX10-NEXT:    workGroupIDX:    { reg: '$sgpr4' }
