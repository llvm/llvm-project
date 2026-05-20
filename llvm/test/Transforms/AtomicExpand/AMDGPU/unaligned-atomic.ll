; RUN: not opt -disable-output -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -passes='require<libcall-lowering-info>,atomic-expand' %s 2>&1 | FileCheck --implicit-check-not=error %s

; CHECK: error: unsupported atomic load: instruction alignment 1 is smaller than the required 4-byte alignment for this atomic operation
define i32 @atomic_load_global_align1(ptr addrspace(1) %ptr) {
  %val = load atomic i32, ptr addrspace(1) %ptr  seq_cst, align 1
  ret i32 %val
}

; CHECK: error: unsupported atomic store: instruction alignment 1 is smaller than the required 4-byte alignment for this atomic operation
define void @atomic_store_global_align1(ptr addrspace(1) %ptr, i32 %val) {
  store atomic i32 %val, ptr addrspace(1) %ptr monotonic, align 1
  ret void
}

; CHECK: error: unsupported atomic load: instruction alignment 2 is smaller than the required 4-byte alignment for this atomic operation
define <2 x half> @atomic_load_v2f16_flat_align2(ptr addrspace(0) %ptr) {
  %val = load atomic <2 x half>, ptr addrspace(0) %ptr syncscope("agent") monotonic, align 2
  ret <2 x half> %val
}

; CHECK: error: unsupported atomic load: instruction alignment 2 is smaller than the required 4-byte alignment for this atomic operation
define <2 x half> @atomic_load_v2f16_global_align2(ptr addrspace(1) %ptr) {
  %val = load atomic <2 x half>, ptr addrspace(1) %ptr syncscope("agent") monotonic, align 2
  ret <2 x half> %val
}

; CHECK: error: unsupported atomic load: instruction alignment 4 is smaller than the required 8-byte alignment for this atomic operation
define <2 x float> @atomic_load_v2f32_acquire_align4(ptr addrspace(1) %ptr) {
  %val = load atomic <2 x float>, ptr addrspace(1) %ptr syncscope("agent") acquire, align 4
  ret <2 x float> %val
}

; CHECK: error: unsupported atomic load: instruction alignment 4 is smaller than the required 8-byte alignment for this atomic operation
define <2 x float> @atomic_load_v2f32_seq_cst_align4(ptr addrspace(1) %ptr) {
  %val = load atomic <2 x float>, ptr addrspace(1) %ptr syncscope("agent") seq_cst, align 4
  ret <2 x float> %val
}

; CHECK: error: unsupported atomic load: instruction alignment 4 is smaller than the required 8-byte alignment for this atomic operation
define <2 x float> @atomic_load_v2f32_lds_align4(ptr addrspace(3) %ptr) {
  %val = load atomic <2 x float>, ptr addrspace(3) %ptr syncscope("agent") monotonic, align 4
  ret <2 x float> %val
}

; CHECK: error: unsupported atomic store: instruction alignment 2 is smaller than the required 4-byte alignment for this atomic operation
define void @atomic_store_v2f16_flat_align2(ptr addrspace(0) %ptr, <2 x half> %val) {
  store atomic <2 x half> %val, ptr addrspace(0) %ptr syncscope("agent") monotonic, align 2
  ret void
}

; CHECK: error: unsupported atomic store: instruction alignment 4 is smaller than the required 8-byte alignment for this atomic operation
define void @atomic_store_v2f32_flat_release_align2(ptr addrspace(0) %ptr, <2 x float> %val) {
  store atomic <2 x float> %val, ptr addrspace(0) %ptr syncscope("agent") release, align 4
  ret void
}

; CHECK: error: unsupported atomic store: instruction alignment 2 is smaller than the required 4-byte alignment for this atomic operation
define void @atomic_store_v2f16_global_align2(ptr addrspace(1) %ptr, <2 x half> %val) {
  store atomic <2 x half> %val, ptr addrspace(1) %ptr syncscope("agent") monotonic, align 2
  ret void
}

; CHECK: error: unsupported atomic store: instruction alignment 4 is smaller than the required 8-byte alignment for this atomic operation
define void @atomic_store_v2f32_lds_align4(ptr addrspace(3) %ptr, <2 x float> %val) {
  store atomic <2 x float> %val, ptr addrspace(3) %ptr syncscope("agent") monotonic, align 4
  ret void
}
