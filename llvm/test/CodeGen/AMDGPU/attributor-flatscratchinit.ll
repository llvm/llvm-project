; Test the generation of the attribute amdgpu-no-flat-scratch-init
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -passes=amdgpu-attributor < %s | FileCheck -check-prefixes=GFX9 %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -passes=amdgpu-attributor < %s | FileCheck -check-prefixes=GFX10 %s

;; tests of addrspacecast

define void @without_global_to_flat_addrspacecast(ptr addrspace(1) %ptr) {
; GFX9-LABEL: define void @without_global_to_flat_addrspacecast(ptr addrspace(1) %ptr)
; GFX9-SAME:  #[[ATTR0_GFX9_NOFSI:[0-9]+]]
;
; GFX10-LABEL: define void @without_global_to_flat_addrspacecast(ptr addrspace(1) %ptr)
; GFX10-SAME:  #[[ATTR0_GFX10_NOFSI:[0-9]+]]
  store volatile i32 0, ptr addrspace(1) %ptr
  ret void
}

define amdgpu_kernel void @without_global_to_flat_addrspacecast_cc_kernel(ptr addrspace(1) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @without_global_to_flat_addrspacecast_cc_kernel(ptr addrspace(1) %ptr)
; GFX9-SAME:  #[[ATTR1_GFX9_NOFSI2:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_kernel void @without_global_to_flat_addrspacecast_cc_kernel(ptr addrspace(1) %ptr)
; GFX10-SAME:  #[[ATTR1_GFX10_NOFSI2:[0-9]+]]
  store volatile i32 0, ptr addrspace(1) %ptr
  ret void
}

define void @with_global_to_flat_addrspacecast(ptr addrspace(1) %ptr) {
; GFX9-LABEL: define void @with_global_to_flat_addrspacecast(ptr addrspace(1) %ptr)
; GFX9-SAME:  #[[ATTR0_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @with_global_to_flat_addrspacecast(ptr addrspace(1) %ptr)
; GFX10-SAME:  #[[ATTR0_GFX10_NOFSI]]
  %stof = addrspacecast ptr addrspace(1) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define amdgpu_kernel void @with_global_to_flat_addrspacecast_cc_kernel(ptr addrspace(1) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @with_global_to_flat_addrspacecast_cc_kernel(ptr addrspace(1) %ptr)
; GFX9-SAME:  #[[ATTR1_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_global_to_flat_addrspacecast_cc_kernel(ptr addrspace(1) %ptr)
; GFX10-SAME:  #[[ATTR1_GFX10_NOFSI2]]
  %stof = addrspacecast ptr addrspace(1) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define void @without_region_to_flat_addrspacecast(ptr addrspace(2) %ptr) {
; GFX9-LABEL: define void @without_region_to_flat_addrspacecast(ptr addrspace(2) %ptr)
; GFX9-SAME:  #[[ATTR0_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @without_region_to_flat_addrspacecast(ptr addrspace(2) %ptr)
; GFX10-SAME:  #[[ATTR0_GFX10_NOFSI]]
  store volatile i32 0, ptr addrspace(2) %ptr
  ret void
}

define amdgpu_kernel void @without_region_to_flat_addrspacecast_cc_kernel(ptr addrspace(2) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @without_region_to_flat_addrspacecast_cc_kernel(ptr addrspace(2) %ptr)
; GFX9-SAME:  #[[ATTR1_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @without_region_to_flat_addrspacecast_cc_kernel(ptr addrspace(2) %ptr)
; GFX10-SAME:  #[[ATTR1_GFX10_NOFSI2]]
  store volatile i32 0, ptr addrspace(2) %ptr
  ret void
}

define void @with_region_to_flat_addrspacecast(ptr addrspace(2) %ptr) {
; GFX9-LABEL: define void @with_region_to_flat_addrspacecast(ptr addrspace(2) %ptr)
; GFX9-SAME:  #[[ATTR0_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @with_region_to_flat_addrspacecast(ptr addrspace(2) %ptr)
; GFX10-SAME:  #[[ATTR0_GFX10_NOFSI]]
  %stof = addrspacecast ptr addrspace(2) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define amdgpu_kernel void @with_region_to_flat_addrspacecast_cc_kernel(ptr addrspace(2) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @with_region_to_flat_addrspacecast_cc_kernel(ptr addrspace(2) %ptr)
; GFX9-SAME:  #[[ATTR1_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_region_to_flat_addrspacecast_cc_kernel(ptr addrspace(2) %ptr)
; GFX10-SAME:  #[[ATTR1_GFX10_NOFSI2]]
  %stof = addrspacecast ptr addrspace(2) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define void @without_group_to_flat_addrspacecast(ptr addrspace(3) %ptr) {
; GFX9-LABEL: define void @without_group_to_flat_addrspacecast(ptr addrspace(3) %ptr)
; GFX9-SAME:  #[[ATTR0_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @without_group_to_flat_addrspacecast(ptr addrspace(3) %ptr)
; GFX10-SAME:  #[[ATTR0_GFX10_NOFSI]]
  store volatile i32 0, ptr addrspace(3) %ptr
  ret void
}

define amdgpu_kernel void @without_group_to_flat_addrspacecast_cc_kernel(ptr addrspace(3) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @without_group_to_flat_addrspacecast_cc_kernel(ptr addrspace(3) %ptr)
; GFX9-SAME:  #[[ATTR1_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @without_group_to_flat_addrspacecast_cc_kernel(ptr addrspace(3) %ptr)
; GFX10-SAME:  #[[ATTR1_GFX10_NOFSI2]]
  store volatile i32 0, ptr addrspace(3) %ptr
  ret void
}

define void @with_group_to_flat_addrspacecast(ptr addrspace(3) %ptr) {
; GFX9-LABEL: define void @with_group_to_flat_addrspacecast(ptr addrspace(3) %ptr)
; GFX9-SAME:  #[[ATTR0_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @with_group_to_flat_addrspacecast(ptr addrspace(3) %ptr)
; GFX10-SAME:  #[[ATTR0_GFX10_NOFSI]]
  %stof = addrspacecast ptr addrspace(3) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define amdgpu_kernel void @with_group_to_flat_addrspacecast_cc_kernel(ptr addrspace(3) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @with_group_to_flat_addrspacecast_cc_kernel(ptr addrspace(3) %ptr)
; GFX9-SAME:  #[[ATTR1_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_group_to_flat_addrspacecast_cc_kernel(ptr addrspace(3) %ptr)
; GFX10-SAME:  #[[ATTR1_GFX10_NOFSI2]]
  %stof = addrspacecast ptr addrspace(3) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define void @without_constant_to_flat_addrspacecast(ptr addrspace(4) %ptr) {
; GFX9-LABEL: define void @without_constant_to_flat_addrspacecast(ptr addrspace(4) %ptr)
; GFX9-SAME:  #[[ATTR0_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @without_constant_to_flat_addrspacecast(ptr addrspace(4) %ptr)
; GFX10-SAME:  #[[ATTR0_GFX10_NOFSI]]
  store volatile i32 0, ptr addrspace(4) %ptr
  ret void
}

define amdgpu_kernel void @without_constant_to_flat_addrspacecast_cc_kernel(ptr addrspace(4) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @without_constant_to_flat_addrspacecast_cc_kernel(ptr addrspace(4) %ptr)
; GFX9-SAME:  #[[ATTR1_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @without_constant_to_flat_addrspacecast_cc_kernel(ptr addrspace(4) %ptr)
; GFX10-SAME:  #[[ATTR1_GFX10_NOFSI2]]
  store volatile i32 0, ptr addrspace(4) %ptr
  ret void
}

define void @with_constant_to_flat_addrspacecast(ptr addrspace(4) %ptr) {
; GFX9-LABEL: define void @with_constant_to_flat_addrspacecast(ptr addrspace(4) %ptr)
; GFX9-SAME:  #[[ATTR0_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @with_constant_to_flat_addrspacecast(ptr addrspace(4) %ptr)
; GFX10-SAME:  #[[ATTR0_GFX10_NOFSI]]
  %stof = addrspacecast ptr addrspace(4) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define amdgpu_kernel void @with_constant_to_flat_addrspacecast_cc_kernel(ptr addrspace(4) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @with_constant_to_flat_addrspacecast_cc_kernel(ptr addrspace(4) %ptr)
; GFX9-SAME:  #[[ATTR1_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_constant_to_flat_addrspacecast_cc_kernel(ptr addrspace(4) %ptr)
; GFX10-SAME:  #[[ATTR1_GFX10_NOFSI2]]
  %stof = addrspacecast ptr addrspace(4) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR0_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR0_GFX10_NOFSI]]
  store volatile i32 0, ptr addrspace(5) %ptr
  ret void
}

define amdgpu_kernel void @without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR1_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR1_GFX10_NOFSI2]]
  store volatile i32 0, ptr addrspace(5) %ptr
  ret void
}

define void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR2_GFX9_NO_NOFSI:[0-9]+]]
;
; GFX10-LABEL: define void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR2_GFX10_NO_NOFSI:[0-9]+]]
  %stof = addrspacecast ptr addrspace(5) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define amdgpu_kernel void @with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR3_GFX9_NO_NOFSI2:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR3_GFX10_NO_NOFSI2:[0-9]+]]
  %stof = addrspacecast ptr addrspace(5) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define void @call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR0_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR0_GFX10_NOFSI]]
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR1_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR1_GFX10_NOFSI2]]
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR2_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR2_GFX10_NO_NOFSI]]
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR3_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR3_GFX10_NO_NOFSI2]]
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR2_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR2_GFX10_NO_NOFSI]]
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_both_with_and_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @call_both_with_and_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR3_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_both_with_and_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR3_GFX10_NO_NOFSI2]]
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @call_call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @call_call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR0_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @call_call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR0_GFX10_NOFSI]]
  call void @call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @call_call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR1_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR1_GFX10_NOFSI2]]
  call void @call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @call_call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @call_call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR2_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @call_call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR2_GFX10_NO_NOFSI]]
  call void @call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @call_call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR3_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR3_GFX10_NO_NOFSI2]]
  call void @call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @call_call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @call_call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR2_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @call_call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR2_GFX10_NO_NOFSI]]
  call void @call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_call_both_with_and_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @call_call_both_with_and_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR3_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_call_both_with_and_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR3_GFX10_NO_NOFSI2]]
  call void @call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @with_cast_call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @with_cast_call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR2_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @with_cast_call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR2_GFX10_NO_NOFSI]]
  %stof = addrspacecast ptr addrspace(5) %ptr to ptr
  store volatile i32 0, ptr %stof
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @with_cast_call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @with_cast_call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR3_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_cast_call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR3_GFX10_NO_NOFSI2]]
  %stof = addrspacecast ptr addrspace(5) %ptr to ptr
  store volatile i32 0, ptr %stof
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @with_cast_call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @with_cast_call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR2_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @with_cast_call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR2_GFX10_NO_NOFSI]]
  %stof = addrspacecast ptr addrspace(5) %ptr to ptr
  store volatile i32 0, ptr %stof
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @with_cast_call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @with_cast_call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR3_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_cast_call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR3_GFX10_NO_NOFSI2]]
  %stof = addrspacecast ptr addrspace(5) %ptr to ptr
  store volatile i32 0, ptr %stof
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

;; tests of addrspacecast in a constant

define amdgpu_kernel void @private_constant_expression_use(ptr addrspace(1) nocapture %out) {
; GFX9-LABEL: define amdgpu_kernel void @private_constant_expression_use(ptr addrspace(1) nocapture %out)
; GFX9-SAME:  #[[ATTR3_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @private_constant_expression_use(ptr addrspace(1) nocapture %out)
; GFX10-SAME:  #[[ATTR3_GFX10_NO_NOFSI2]]
  store volatile ptr addrspacecast (ptr addrspace(5) inttoptr (i32 123 to ptr addrspace(5)) to ptr), ptr addrspace(1) %out, align 8
  ret void
}

;; tests of indirect call, intrinsics, inline asm

@gv.fptr0 = external hidden unnamed_addr addrspace(4) constant ptr, align 4

define void @with_indirect_call() {
; GFX9-LABEL: define void @with_indirect_call()
; GFX9-SAME:  #[[ATTR2_GFX9_IND_CALL:[0-9]+]]
;
; GFX10-LABEL: define void @with_indirect_call()
; GFX10-SAME:  #[[ATTR2_GFX10_IND_CALL:[0-9]+]] {
  %fptr = load ptr, ptr addrspace(4) @gv.fptr0
  call void %fptr()
  ret void
}

define amdgpu_kernel void @with_indirect_call_cc_kernel() {
; GFX9-LABEL: define amdgpu_kernel void @with_indirect_call_cc_kernel()
; GFX9-SAME:  #[[ATTR3_GFX9_IND_CALL2:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_indirect_call_cc_kernel()
; GFX10-SAME:  #[[ATTR3_GFX10_IND_CALL2:[0-9]+]]
  %fptr = load ptr, ptr addrspace(4) @gv.fptr0
  call void %fptr()
  ret void
}

define void @call_with_indirect_call() {
; GFX9-LABEL: define void @call_with_indirect_call()
; GFX9-SAME:  #[[ATTR4_GFX9_IND_CALL:[0-9]+]]
;
; GFX10-LABEL: define void @call_with_indirect_call()
; GFX10-SAME:  #[[ATTR4_GFX10_IND_CALL:[0-9]+]]
  call void @with_indirect_call()
  ret void
}

define amdgpu_kernel void @call_with_indirect_call_cc_kernel() {
; GFX9-LABEL: define amdgpu_kernel void @call_with_indirect_call_cc_kernel()
; GFX9-SAME:  #[[ATTR5_GFX9_IND_CALL2:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_with_indirect_call_cc_kernel()
; GFX10-SAME:  #[[ATTR5_GFX10_IND_CALL2:[0-9]+]]
  call void @with_indirect_call()
  ret void
}

define void @empty() {
  ret void
}

define void @also_empty() {
  ret void
}

define amdgpu_kernel void @indirect_call_known_callees(i1 %cond) {
; GFX9-LABEL: define amdgpu_kernel void @indirect_call_known_callees(i1 %cond)
; GFX9-SAME:  #[[ATTR6_GFX9_NOFSI3:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_kernel void @indirect_call_known_callees(i1 %cond)
; GFX10-SAME:  #[[ATTR6_GFX10_NOFSI3:[0-9]+]]
  %fptr = select i1 %cond, ptr @empty, ptr @also_empty
  call void %fptr()
  ret void
}

declare i32 @llvm.amdgcn.workgroup.id.x()

define void @use_intrinsic_workitem_id_x() {
; GFX9-LABEL: define void @use_intrinsic_workitem_id_x()
; GFX9-SAME:  #[[ATTR8_GFX9_NOFSI4:[0-9]+]]
;
; GFX10-LABEL: define void @use_intrinsic_workitem_id_x()
; GFX10-SAME:  #[[ATTR8_GFX10_NOFSI4:[0-9]+]]
  %val = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %val, ptr addrspace(1) null
  ret void
}

define amdgpu_kernel void @use_intrinsic_workitem_id_x_cc_kernel() {
; GFX9-LABEL: define amdgpu_kernel void @use_intrinsic_workitem_id_x_cc_kernel()
; GFX9-SAME:  #[[ATTR1_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @use_intrinsic_workitem_id_x_cc_kernel()
; GFX10-SAME:  #[[ATTR1_GFX10_NOFSI2]]
  %val = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %val, ptr addrspace(1) null
  ret void
}

define void @call_use_intrinsic_workitem_id_x() {
; GFX9-LABEL: define void @call_use_intrinsic_workitem_id_x()
; GFX9-SAME:  #[[ATTR6_GFX9_NOFSI4:[0-9]+]]
;
; GFX10-LABEL: define void @call_use_intrinsic_workitem_id_x()
; GFX10-SAME:  #[[ATTR6_GFX10_NOFSI4:[0-9]+]]
  call void @use_intrinsic_workitem_id_x()
  ret void
}

define amdgpu_kernel void @call_use_intrinsic_workitem_id_x_cc_kernel() {
; GFX9-LABEL: define amdgpu_kernel void @call_use_intrinsic_workitem_id_x_cc_kernel()
; GFX9-SAME:  #[[ATTR9_GFX9_NOFSI5:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_use_intrinsic_workitem_id_x_cc_kernel()
; GFX10-SAME:  #[[ATTR9_GFX10_NOFSI5:[0-9]+]]
  call void @use_intrinsic_workitem_id_x()
  ret void
}

define amdgpu_kernel void @calls_intrin_ascast_cc_kernel(ptr addrspace(3) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @calls_intrin_ascast_cc_kernel(ptr addrspace(3) %ptr)
; GFX9-SAME:  #[[ATTR3_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @calls_intrin_ascast_cc_kernel(ptr addrspace(3) %ptr)
; GFX10-SAME:  #[[ATTR3_GFX10_NO_NOFSI2]]
  %1 = call ptr @llvm.amdgcn.addrspacecast.nonnull.p0.p3(ptr addrspace(3) %ptr)
  store volatile i32 7, ptr %1, align 4
  ret void
}

define amdgpu_kernel void @call_calls_intrin_ascast_cc_kernel(ptr addrspace(3) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @call_calls_intrin_ascast_cc_kernel(ptr addrspace(3) %ptr)
; GFX9-SAME:  #[[ATTR3_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_calls_intrin_ascast_cc_kernel(ptr addrspace(3) %ptr)
; GFX10-SAME:  #[[ATTR3_GFX10_NO_NOFSI2]]
  call void @calls_intrin_ascast_cc_kernel(ptr addrspace(3) %ptr)
  ret void
}

define amdgpu_kernel void @with_inline_asm() {
; GFX9-LABEL: with_inline_asm
; GFX9-SAME:  #[[ATTR6_GFX9_NOFSI3]]
;
; GFX10-LABEL: with_inline_asm
; GFX10-SAME:  #[[ATTR6_GFX10_NOFSI3]]
  call void asm sideeffect "; use $0", "a"(i32 poison)
  ret void
}

; GFX9:  attributes #[[ATTR0_GFX9_NOFSI]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="4,10" "target-cpu"="gfx900" "uniform-work-group-size"="false" }

; GFX9:  attributes #[[ATTR1_GFX9_NOFSI2]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "target-cpu"="gfx900" "uniform-work-group-size"="false" }

; GFX9:  attributes #[[ATTR2_GFX9_NO_NOFSI]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="4,10" "target-cpu"="gfx900" "uniform-work-group-size"="false" }

; GFX9:  attributes #[[ATTR3_GFX9_NO_NOFSI2]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "target-cpu"="gfx900" "uniform-work-group-size"="false" }

; GFX9:  attributes #[[ATTR4_GFX9_IND_CALL]] = { "amdgpu-waves-per-eu"="4,10" "target-cpu"="gfx900" "uniform-work-group-size"="false" }
; GFX9:  attributes #[[ATTR5_GFX9_IND_CALL2]] = { "target-cpu"="gfx900" "uniform-work-group-size"="false" }

; GFX9:  attributes #[[ATTR6_GFX9_NOFSI3]] = { "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "target-cpu"="gfx900" "uniform-work-group-size"="false" }

; GFX9:  attributes #[[ATTR8_GFX9_NOFSI4]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="4,10" "target-cpu"="gfx900" "uniform-work-group-size"="false" }

; GFX9:  attributes #[[ATTR9_GFX9_NOFSI5]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "target-cpu"="gfx900" "uniform-work-group-size"="false" }






; GFX10:  attributes #[[ATTR0_GFX10_NOFSI]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="8,20" "target-cpu"="gfx1010" "uniform-work-group-size"="false" }

; GFX10:  attributes #[[ATTR1_GFX10_NOFSI2]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "target-cpu"="gfx1010" "uniform-work-group-size"="false" }

; GFX10:  attributes #[[ATTR2_GFX10_NO_NOFSI]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="8,20" "target-cpu"="gfx1010" "uniform-work-group-size"="false" }

; GFX10:  attributes #[[ATTR3_GFX10_NO_NOFSI2]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "target-cpu"="gfx1010" "uniform-work-group-size"="false" }

; GFX10:  attributes #[[ATTR4_GFX10_IND_CALL]] = { "amdgpu-waves-per-eu"="8,20" "target-cpu"="gfx1010" "uniform-work-group-size"="false" }
; GFX10:  attributes #[[ATTR5_GFX10_IND_CALL2]] = { "target-cpu"="gfx1010" "uniform-work-group-size"="false" }

; GFX10:  attributes #[[ATTR6_GFX10_NOFSI3]] = { "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "target-cpu"="gfx1010" "uniform-work-group-size"="false" }

; GFX10:  attributes #[[ATTR8_GFX10_NOFSI4]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="8,20" "target-cpu"="gfx1010" "uniform-work-group-size"="false" }

; GFX10:  attributes #[[ATTR9_GFX10_NOFSI5]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "target-cpu"="gfx1010" "uniform-work-group-size"="false" }
