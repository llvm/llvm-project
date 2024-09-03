; Test the generation of the attribute amdgpu-no-flat-scratch-init
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -passes=amdgpu-attributor < %s | FileCheck -check-prefixes=GFX9 %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -passes=amdgpu-attributor < %s | FileCheck -check-prefixes=GFX10 %s

;; tests of alloca

define void @without_alloca(i1 %arg0) {
; GFX9-LABEL: define void @without_alloca(i1 %arg0)
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI:[0-9]+]]
;
; GFX10-LABEL: define void @without_alloca(i1 %arg0)
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI:[0-9]+]]
  store volatile i1 %arg0, ptr addrspace(1) null
  ret void
}

define void @with_alloca() {
; GFX9-LABEL: define void @with_alloca()
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI:[0-9]+]]
;
; GFX10-LABEL: define void @with_alloca()
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI:[0-9]+]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  ret void
}

define amdgpu_kernel void @without_alloca_cc_kernel(i1 %arg0) {
; GFX9-LABEL: define amdgpu_kernel void @without_alloca_cc_kernel(i1 %arg0)
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI2:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_kernel void @without_alloca_cc_kernel(i1 %arg0)
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI2:[0-9]+]]
  store volatile i1 %arg0, ptr addrspace(1) null
  ret void
}

define amdgpu_kernel void @with_alloca_cc_kernel() {
; GFX9-LABEL: define amdgpu_kernel void @with_alloca_cc_kernel()
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_alloca_cc_kernel()
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2:[0-9]+]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  ret void
}

; graphics functions won't get the attribute amdgpu-no-flat-scratch-init

define amdgpu_vs void @with_alloca_cc_vs() {
; GFX9-LABEL: define amdgpu_vs void @with_alloca_cc_vs()
; GFX9-SAME:  #[[ATTR_GFX9_CC_GRAPHICS:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_vs void @with_alloca_cc_vs()
; GFX10-SAME:  #[[ATTR_GFX10_CC_GRAPHICS:[0-9]+]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  ret void
}

define amdgpu_gs void @with_alloca_cc_gs() {
; GFX9-LABEL: define amdgpu_gs void @with_alloca_cc_gs()
; GFX9-SAME:  #[[ATTR_GFX9_CC_GRAPHICS:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_gs void @with_alloca_cc_gs()
; GFX10-SAME:  #[[ATTR_GFX10_CC_GRAPHICS:[0-9]+]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  ret void
}

define amdgpu_ps void @with_alloca_cc_ps() {
; GFX9-LABEL: define amdgpu_ps void @with_alloca_cc_ps()
; GFX9-SAME:  #[[ATTR_GFX9_CC_GRAPHICS:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_ps void @with_alloca_cc_ps()
; GFX10-SAME:  #[[ATTR_GFX10_CC_GRAPHICS:[0-9]+]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  ret void
}

define amdgpu_cs void @with_alloca_cc_cs() {
; GFX9-LABEL: define amdgpu_cs void @with_alloca_cc_cs()
; GFX9-SAME:  #[[ATTR_GFX9_CC_GRAPHICS:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_cs void @with_alloca_cc_cs()
; GFX10-SAME:  #[[ATTR_GFX10_CC_GRAPHICS:[0-9]+]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  ret void
}

define amdgpu_hs void @with_alloca_cc_hs() {
; GFX9-LABEL: define amdgpu_hs void @with_alloca_cc_hs()
; GFX9-SAME:  #[[ATTR_GFX9_CC_GRAPHICS:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_hs void @with_alloca_cc_hs()
; GFX10-SAME:  #[[ATTR_GFX10_CC_GRAPHICS:[0-9]+]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  ret void
}

define amdgpu_ls void @with_alloca_cc_ls() {
; GFX9-LABEL: define amdgpu_ls void @with_alloca_cc_ls()
; GFX9-SAME:  #[[ATTR_GFX9_CC_GRAPHICS:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_ls void @with_alloca_cc_ls()
; GFX10-SAME:  #[[ATTR_GFX10_CC_GRAPHICS:[0-9]+]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  ret void
}

define amdgpu_es void @with_alloca_cc_es() {
; GFX9-LABEL: define amdgpu_es void @with_alloca_cc_es()
; GFX9-SAME:  #[[ATTR_GFX9_CC_GRAPHICS:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_es void @with_alloca_cc_es()
; GFX10-SAME:  #[[ATTR_GFX10_CC_GRAPHICS:[0-9]+]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  ret void
}

define amdgpu_gfx void @with_alloca_cc_gfx() {
; GFX9-LABEL: define amdgpu_gfx void @with_alloca_cc_gfx()
; GFX9-SAME:  #[[ATTR_GFX9_CC_GRAPHICS2:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_gfx void @with_alloca_cc_gfx()
; GFX10-SAME:  #[[ATTR_GFX10_CC_GRAPHICS2:[0-9]+]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  ret void
}

define amdgpu_cs_chain void @with_alloca_cc_cs_chain() {
; GFX9-LABEL: define amdgpu_cs_chain void @with_alloca_cc_cs_chain()
; GFX9-SAME:  #[[ATTR_GFX9_CC_GRAPHICS2:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_cs_chain void @with_alloca_cc_cs_chain()
; GFX10-SAME:  #[[ATTR_GFX10_CC_GRAPHICS2:[0-9]+]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  ret void
}

define amdgpu_cs_chain_preserve void @with_alloca_cc_cs_chain_preserve() {
; GFX9-LABEL: define amdgpu_cs_chain_preserve void @with_alloca_cc_cs_chain_preserve()
; GFX9-SAME:  #[[ATTR_GFX9_CC_GRAPHICS2:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_cs_chain_preserve void @with_alloca_cc_cs_chain_preserve()
; GFX10-SAME:  #[[ATTR_GFX10_CC_GRAPHICS2:[0-9]+]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  ret void
}

define void @call_without_alloca() {
; GFX9-LABEL: define void @call_without_alloca()
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @call_without_alloca()
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI]]
  call void @without_alloca(i1 true)
  ret void
}

define amdgpu_kernel void @call_without_alloca_cc_kernel() {
; GFX9-LABEL: define amdgpu_kernel void @call_without_alloca_cc_kernel()
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_without_alloca_cc_kernel()
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI2]]
  call void @without_alloca(i1 true)
  ret void
}

define void @call_with_alloca() {
; GFX9-LABEL: define void @call_with_alloca()
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @call_with_alloca()
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI]]
  call void @with_alloca()
  ret void
}

define amdgpu_kernel void @call_with_alloca_cc_kernel() {
; GFX9-LABEL: define amdgpu_kernel void @call_with_alloca_cc_kernel()
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_with_alloca_cc_kernel()
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  call void @with_alloca()
  ret void
}

define void @call_both_with_and_without_alloca() {
; GFX9-LABEL: define void @call_both_with_and_without_alloca()
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI3:[0-9]+]]
;
; GFX10-LABEL: define void @call_both_with_and_without_alloca()
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI3:[0-9]+]]
  call void @with_alloca()
  call void @without_alloca()
  ret void
}

define amdgpu_kernel void @call_both_with_and_without_alloca_cc_kernel() {
; GFX9-LABEL: define amdgpu_kernel void @call_both_with_and_without_alloca_cc_kernel()
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI4:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_both_with_and_without_alloca_cc_kernel()
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI4:[0-9]+]]
  call void @with_alloca()
  call void @without_alloca()
  ret void
}

define void @call_call_without_alloca() {
; GFX9-LABEL: define void @call_call_without_alloca()
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @call_call_without_alloca()
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI]]
  call void @call_without_alloca()
  ret void
}

define amdgpu_kernel void @call_call_without_alloca_cc_kernel() {
; GFX9-LABEL: define amdgpu_kernel void @call_call_without_alloca_cc_kernel()
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_call_without_alloca_cc_kernel()
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI2]]
  call void @call_without_alloca()
  ret void
}

define void @call_call_with_alloca() {
; GFX9-LABEL: define void @call_call_with_alloca()
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @call_call_with_alloca()
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI]]
  call void @call_with_alloca()
  ret void
}

define amdgpu_kernel void @call_call_with_alloca_cc_kernel() {
; GFX9-LABEL: define amdgpu_kernel void @call_call_with_alloca_cc_kernel()
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_call_with_alloca_cc_kernel()
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  call void @call_with_alloca()
  ret void
}

define void @with_alloca_call_without_alloca() {
; GFX9-LABEL: define void @with_alloca_call_without_alloca()
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @with_alloca_call_without_alloca()
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  call void @without_alloca()
  ret void
}

define amdgpu_kernel void @with_alloca_call_without_alloca_cc_kernel() {
; GFX9-LABEL: define amdgpu_kernel void @with_alloca_call_without_alloca_cc_kernel()
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_alloca_call_without_alloca_cc_kernel()
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  call void @without_alloca()
  ret void
}

define void @with_alloca_call_with_alloca() {
; GFX9-LABEL: define void @with_alloca_call_with_alloca()
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @with_alloca_call_with_alloca()
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  call void @with_alloca()
  ret void
}

define amdgpu_kernel void @with_alloca_call_with_alloca_cc_kernel() {
; GFX9-LABEL: define amdgpu_kernel void @with_alloca_call_with_alloca_cc_kernel()
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_alloca_call_with_alloca_cc_kernel()
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  call void @with_alloca()
  ret void
}

define void @with_alloca_call_call_without_alloca() {
; GFX9-LABEL: define void @with_alloca_call_call_without_alloca()
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @with_alloca_call_call_without_alloca()
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  call void @call_without_alloca()
  ret void
}

define amdgpu_kernel void @with_alloca_call_call_without_alloca_cc_kernel() {
; GFX9-LABEL: define amdgpu_kernel void @with_alloca_call_call_without_alloca_cc_kernel()
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_alloca_call_call_without_alloca_cc_kernel()
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  call void @call_without_alloca()
  ret void
}

define void @with_alloca_call_call_with_alloca() {
; GFX9-LABEL: define void @with_alloca_call_call_with_alloca()
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @with_alloca_call_call_with_alloca()
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  call void @call_with_alloca()
  ret void
}

define amdgpu_kernel void @with_alloca_call_call_with_alloca_cc_kernel() {
; GFX9-LABEL: define amdgpu_kernel void @with_alloca_call_call_with_alloca_cc_kernel()
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_alloca_call_call_with_alloca_cc_kernel()
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  %temp = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 4
  call void @call_with_alloca()
  ret void
}

;; tests of addrspacecast

define void @without_global_to_flat_addrspacecast(ptr addrspace(1) %ptr) {
; GFX9-LABEL: define void @without_global_to_flat_addrspacecast(ptr addrspace(1) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @without_global_to_flat_addrspacecast(ptr addrspace(1) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI]]
  store volatile i32 0, ptr addrspace(1) %ptr
  ret void
}

define amdgpu_kernel void @without_global_to_flat_addrspacecast_cc_kernel(ptr addrspace(1) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @without_global_to_flat_addrspacecast_cc_kernel(ptr addrspace(1) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @without_global_to_flat_addrspacecast_cc_kernel(ptr addrspace(1) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI2]]
  store volatile i32 0, ptr addrspace(1) %ptr
  ret void
}

define void @with_global_to_flat_addrspacecast(ptr addrspace(1) %ptr) {
; GFX9-LABEL: define void @with_global_to_flat_addrspacecast(ptr addrspace(1) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @with_global_to_flat_addrspacecast(ptr addrspace(1) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI]]
  %stof = addrspacecast ptr addrspace(1) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define amdgpu_kernel void @with_global_to_flat_addrspacecast_cc_kernel(ptr addrspace(1) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @with_global_to_flat_addrspacecast_cc_kernel(ptr addrspace(1) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_global_to_flat_addrspacecast_cc_kernel(ptr addrspace(1) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  %stof = addrspacecast ptr addrspace(1) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define void @without_region_to_flat_addrspacecast(ptr addrspace(2) %ptr) {
; GFX9-LABEL: define void @without_region_to_flat_addrspacecast(ptr addrspace(2) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @without_region_to_flat_addrspacecast(ptr addrspace(2) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI]]
  store volatile i32 0, ptr addrspace(2) %ptr
  ret void
}

define amdgpu_kernel void @without_region_to_flat_addrspacecast_cc_kernel(ptr addrspace(2) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @without_region_to_flat_addrspacecast_cc_kernel(ptr addrspace(2) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @without_region_to_flat_addrspacecast_cc_kernel(ptr addrspace(2) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI2]]
  store volatile i32 0, ptr addrspace(2) %ptr
  ret void
}

define void @with_region_to_flat_addrspacecast(ptr addrspace(2) %ptr) {
; GFX9-LABEL: define void @with_region_to_flat_addrspacecast(ptr addrspace(2) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @with_region_to_flat_addrspacecast(ptr addrspace(2) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI]]
  %stof = addrspacecast ptr addrspace(2) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define amdgpu_kernel void @with_region_to_flat_addrspacecast_cc_kernel(ptr addrspace(2) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @with_region_to_flat_addrspacecast_cc_kernel(ptr addrspace(2) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_region_to_flat_addrspacecast_cc_kernel(ptr addrspace(2) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  %stof = addrspacecast ptr addrspace(2) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define void @without_group_to_flat_addrspacecast(ptr addrspace(3) %ptr) {
; GFX9-LABEL: define void @without_group_to_flat_addrspacecast(ptr addrspace(3) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @without_group_to_flat_addrspacecast(ptr addrspace(3) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI]]
  store volatile i32 0, ptr addrspace(3) %ptr
  ret void
}

define amdgpu_kernel void @without_group_to_flat_addrspacecast_cc_kernel(ptr addrspace(3) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @without_group_to_flat_addrspacecast_cc_kernel(ptr addrspace(3) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @without_group_to_flat_addrspacecast_cc_kernel(ptr addrspace(3) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI2]]
  store volatile i32 0, ptr addrspace(3) %ptr
  ret void
}

define void @with_group_to_flat_addrspacecast(ptr addrspace(3) %ptr) {
; GFX9-LABEL: define void @with_group_to_flat_addrspacecast(ptr addrspace(3) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @with_group_to_flat_addrspacecast(ptr addrspace(3) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI]]
  %stof = addrspacecast ptr addrspace(3) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define amdgpu_kernel void @with_group_to_flat_addrspacecast_cc_kernel(ptr addrspace(3) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @with_group_to_flat_addrspacecast_cc_kernel(ptr addrspace(3) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_group_to_flat_addrspacecast_cc_kernel(ptr addrspace(3) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  %stof = addrspacecast ptr addrspace(3) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define void @without_constant_to_flat_addrspacecast(ptr addrspace(4) %ptr) {
; GFX9-LABEL: define void @without_constant_to_flat_addrspacecast(ptr addrspace(4) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @without_constant_to_flat_addrspacecast(ptr addrspace(4) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI]]
  store volatile i32 0, ptr addrspace(4) %ptr
  ret void
}

define amdgpu_kernel void @without_constant_to_flat_addrspacecast_cc_kernel(ptr addrspace(4) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @without_constant_to_flat_addrspacecast_cc_kernel(ptr addrspace(4) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @without_constant_to_flat_addrspacecast_cc_kernel(ptr addrspace(4) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI2]]
  store volatile i32 0, ptr addrspace(4) %ptr
  ret void
}

define void @with_constant_to_flat_addrspacecast(ptr addrspace(4) %ptr) {
; GFX9-LABEL: define void @with_constant_to_flat_addrspacecast(ptr addrspace(4) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @with_constant_to_flat_addrspacecast(ptr addrspace(4) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI]]
  %stof = addrspacecast ptr addrspace(4) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define amdgpu_kernel void @with_constant_to_flat_addrspacecast_cc_kernel(ptr addrspace(4) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @with_constant_to_flat_addrspacecast_cc_kernel(ptr addrspace(4) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_constant_to_flat_addrspacecast_cc_kernel(ptr addrspace(4) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  %stof = addrspacecast ptr addrspace(4) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI]]
  store volatile i32 0, ptr addrspace(5) %ptr
  ret void
}

define amdgpu_kernel void @without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI2]]
  store volatile i32 0, ptr addrspace(5) %ptr
  ret void
}

define void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI]]
  %stof = addrspacecast ptr addrspace(5) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define amdgpu_kernel void @with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  %stof = addrspacecast ptr addrspace(5) %ptr to ptr
  store volatile i32 0, ptr %stof
  ret void
}

define void @call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI]]
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI2]]
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI]]
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI]]
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_both_with_and_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @call_both_with_and_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_both_with_and_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @call_call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @call_call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @call_call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI]]
  call void @call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @call_call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI2]]
  call void @call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @call_call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @call_call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @call_call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI]]
  call void @call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @call_call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  call void @call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @call_call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @call_call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @call_call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI]]
  call void @call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_call_both_with_and_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @call_call_both_with_and_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_call_both_with_and_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  call void @call_both_with_and_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @with_cast_call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @with_cast_call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @with_cast_call_without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI]]
  %stof = addrspacecast ptr addrspace(5) %ptr to ptr
  store volatile i32 0, ptr %stof
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @with_cast_call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @with_cast_call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_cast_call_without_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  %stof = addrspacecast ptr addrspace(5) %ptr to ptr
  store volatile i32 0, ptr %stof
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @with_cast_call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @with_cast_call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @with_cast_call_with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI]]
  %stof = addrspacecast ptr addrspace(5) %ptr to ptr
  store volatile i32 0, ptr %stof
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @with_cast_call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @with_cast_call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_cast_call_with_private_to_flat_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  %stof = addrspacecast ptr addrspace(5) %ptr to ptr
  store volatile i32 0, ptr %stof
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

;; tests of mixed alloca and addrspacecast

define void @call_without_alloca_and_without_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @call_without_alloca_and_without_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI]]
;
; GFX10-LABEL: define void @call_without_alloca_and_without_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI]]
  call void @without_alloca(i1 true)
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_without_alloca_and_without_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @call_without_alloca_and_without_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_without_alloca_and_without_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI2]]
  call void @without_alloca(i1 true)
  call void @without_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define void @call_without_alloca_and_with_addrspacecast(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define void @call_without_alloca_and_with_addrspacecast(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI]]
;
; GFX10-LABEL: define void @call_without_alloca_and_with_addrspacecast(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI]]
  call void @without_alloca(i1 true)
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

define amdgpu_kernel void @call_without_alloca_and_with_addrspacecast_cc_kernel(ptr addrspace(5) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @call_without_alloca_and_with_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_without_alloca_and_with_addrspacecast_cc_kernel(ptr addrspace(5) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  call void @without_alloca(i1 true)
  call void @with_private_to_flat_addrspacecast(ptr addrspace(5) %ptr)
  ret void
}

;; tests of indirect call, intrinsics, inline asm

@gv.fptr0 = external hidden unnamed_addr addrspace(4) constant ptr, align 4

define void @with_indirect_call() {
; GFX9-LABEL: define void @with_indirect_call()
; GFX9-SAME:  #[[ATTR_GFX9_IND_CALL:[0-9]+]]
;
; GFX10-LABEL: define void @with_indirect_call()
; GFX10-SAME:  #[[ATTR_GFX10_IND_CALL:[0-9]+]] {
  %fptr = load ptr, ptr addrspace(4) @gv.fptr0
  call void %fptr()
  ret void
}

define amdgpu_kernel void @with_indirect_call_cc_kernel() {
; GFX9-LABEL: define amdgpu_kernel void @with_indirect_call_cc_kernel()
; GFX9-SAME:  #[[ATTR_GFX9_IND_CALL2:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_kernel void @with_indirect_call_cc_kernel()
; GFX10-SAME:  #[[ATTR_GFX10_IND_CALL2:[0-9]+]]
  %fptr = load ptr, ptr addrspace(4) @gv.fptr0
  call void %fptr()
  ret void
}

define void @call_with_indirect_call() {
; GFX9-LABEL: define void @call_with_indirect_call()
; GFX9-SAME:  #[[ATTR_GFX9_IND_CALL:[0-9]+]]
;
; GFX10-LABEL: define void @call_with_indirect_call()
; GFX10-SAME:  #[[ATTR_GFX10_IND_CALL:[0-9]+]]
  call void @with_indirect_call()
  ret void
}

define amdgpu_kernel void @call_with_indirect_call_cc_kernel() {
; GFX9-LABEL: define amdgpu_kernel void @call_with_indirect_call_cc_kernel()
; GFX9-SAME:  #[[ATTR_GFX9_IND_CALL2:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_with_indirect_call_cc_kernel()
; GFX10-SAME:  #[[ATTR_GFX10_IND_CALL2:[0-9]+]]
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
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI3:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_kernel void @indirect_call_known_callees(i1 %cond)
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI3:[0-9]+]]
  %fptr = select i1 %cond, ptr @empty, ptr @also_empty
  call void %fptr()
  ret void
}

declare i32 @llvm.amdgcn.workgroup.id.x()

define void @use_intrinsic_workitem_id_x() {
; GFX9-LABEL: define void @use_intrinsic_workitem_id_x()
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI4:[0-9]+]]
;
; GFX10-LABEL: define void @use_intrinsic_workitem_id_x()
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI4:[0-9]+]]
  %val = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %val, ptr addrspace(1) null
  ret void
}

define amdgpu_kernel void @use_intrinsic_workitem_id_x_cc_kernel() {
; GFX9-LABEL: define amdgpu_kernel void @use_intrinsic_workitem_id_x_cc_kernel()
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @use_intrinsic_workitem_id_x_cc_kernel()
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI2]]
  %val = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %val, ptr addrspace(1) null
  ret void
}

define void @call_use_intrinsic_workitem_id_x() {
; GFX9-LABEL: define void @call_use_intrinsic_workitem_id_x()
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI4]]
;
; GFX10-LABEL: define void @call_use_intrinsic_workitem_id_x()
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI4]]
  call void @use_intrinsic_workitem_id_x()
  ret void
}

define amdgpu_kernel void @call_use_intrinsic_workitem_id_x_cc_kernel() {
; GFX9-LABEL: define amdgpu_kernel void @call_use_intrinsic_workitem_id_x_cc_kernel()
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI5:[0-9]+]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_use_intrinsic_workitem_id_x_cc_kernel()
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI5:[0-9]+]]
  call void @use_intrinsic_workitem_id_x()
  ret void
}

define amdgpu_kernel void @calls_intrin_ascast_cc_kernel(ptr addrspace(3) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @calls_intrin_ascast_cc_kernel(ptr addrspace(3) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @calls_intrin_ascast_cc_kernel(ptr addrspace(3) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  %1 = call ptr @llvm.amdgcn.addrspacecast.nonnull.p0.p3(ptr addrspace(3) %ptr)
  store volatile i32 7, ptr %1, align 4
  ret void
}

define amdgpu_kernel void @call_calls_intrin_ascast_cc_kernel(ptr addrspace(3) %ptr) {
; GFX9-LABEL: define amdgpu_kernel void @call_calls_intrin_ascast_cc_kernel(ptr addrspace(3) %ptr)
; GFX9-SAME:  #[[ATTR_GFX9_NO_NOFSI2]]
;
; GFX10-LABEL: define amdgpu_kernel void @call_calls_intrin_ascast_cc_kernel(ptr addrspace(3) %ptr)
; GFX10-SAME:  #[[ATTR_GFX10_NO_NOFSI2]]
  call void @calls_intrin_ascast_cc_kernel(ptr addrspace(3) %ptr)
  ret void
}

define amdgpu_kernel void @with_inline_asm() {
; GFX9-LABEL: with_inline_asm
; GFX9-SAME:  #[[ATTR_GFX9_NOFSI3]]
;
; GFX10-LABEL: with_inline_asm
; GFX10-SAME:  #[[ATTR_GFX10_NOFSI3]]
  call void asm sideeffect "; use $0", "a"(i32 poison)
  ret void
}

; GFX9:  attributes #[[ATTR_GFX9_NOFSI]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="4,10" "target-cpu"="gfx900" "uniform-work-group-size"="false" }

; GFX9:  attributes #[[ATTR_GFX9_NO_NOFSI]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="4,10" "target-cpu"="gfx900" "uniform-work-group-size"="false" }

; GFX9:  attributes #[[ATTR_GFX9_NOFSI2]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "target-cpu"="gfx900" "uniform-work-group-size"="false" }

; GFX9:  attributes #[[ATTR_GFX9_NO_NOFSI2]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "target-cpu"="gfx900" "uniform-work-group-size"="false" }

; GFX9:  attributes #[[ATTR_GFX9_CC_GRAPHICS]] = { "amdgpu-no-agpr" "target-cpu"="gfx900" "uniform-work-group-size"="false" }
; GFX9:  attributes #[[ATTR_GFX9_CC_GRAPHICS2]] = { "amdgpu-no-agpr" "amdgpu-waves-per-eu"="4,10" "target-cpu"="gfx900" "uniform-work-group-size"="false" }

; GFX9:  attributes #[[ATTR_GFX9_IND_CALL]] = { "amdgpu-waves-per-eu"="4,10" "target-cpu"="gfx900" "uniform-work-group-size"="false" }
; GFX9:  attributes #[[ATTR_GFX9_IND_CALL2]] = { "target-cpu"="gfx900" "uniform-work-group-size"="false" }

; GFX9:  attributes #[[ATTR_GFX9_NOFSI3]] = { "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "target-cpu"="gfx900" "uniform-work-group-size"="false" }

; GFX9:  attributes #[[ATTR_GFX9_NOFSI4]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="4,10" "target-cpu"="gfx900" "uniform-work-group-size"="false" }

; GFX9:  attributes #[[ATTR_GFX9_NOFSI5]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "target-cpu"="gfx900" "uniform-work-group-size"="false" }






; GFX10:  attributes #[[ATTR_GFX10_NOFSI]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="8,20" "target-cpu"="gfx1010" "uniform-work-group-size"="false" }

; GFX10:  attributes #[[ATTR_GFX10_NO_NOFSI]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="8,20" "target-cpu"="gfx1010" "uniform-work-group-size"="false" }

; GFX10:  attributes #[[ATTR_GFX10_NOFSI2]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "target-cpu"="gfx1010" "uniform-work-group-size"="false" }

; GFX10:  attributes #[[ATTR_GFX10_NO_NOFSI2]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "target-cpu"="gfx1010" "uniform-work-group-size"="false" }

; GFX10:  attributes #[[ATTR_GFX10_CC_GRAPHICS]] = { "amdgpu-no-agpr" "target-cpu"="gfx1010" "uniform-work-group-size"="false" }
; GFX10:  attributes #[[ATTR_GFX10_CC_GRAPHICS2]] = { "amdgpu-no-agpr" "amdgpu-waves-per-eu"="8,20" "target-cpu"="gfx1010" "uniform-work-group-size"="false" }

; GFX10:  attributes #[[ATTR_GFX10_IND_CALL]] = { "amdgpu-waves-per-eu"="8,20" "target-cpu"="gfx1010" "uniform-work-group-size"="false" }
; GFX10:  attributes #[[ATTR_GFX10_IND_CALL2]] = { "target-cpu"="gfx1010" "uniform-work-group-size"="false" }

; GFX10:  attributes #[[ATTR_GFX10_NOFSI3]] = { "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "target-cpu"="gfx1010" "uniform-work-group-size"="false" }

; GFX10:  attributes #[[ATTR_GFX10_NOFSI4]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="8,20" "target-cpu"="gfx1010" "uniform-work-group-size"="false" }

; GFX10:  attributes #[[ATTR_GFX10_NOFSI5]] = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "target-cpu"="gfx1010" "uniform-work-group-size"="false" }
