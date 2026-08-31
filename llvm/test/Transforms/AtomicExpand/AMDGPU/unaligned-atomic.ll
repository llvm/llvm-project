; RUN: not opt -disable-output -mtriple=amdgpu9.00-amd-amdhsa -passes='require<libcall-lowering-info>,atomic-expand' %s 2>&1 | FileCheck --implicit-check-not=error %s

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

; CHECK: error: unsupported atomic load: instruction alignment 2 is smaller than the required 8-byte alignment for this atomic operation
define <2 x float> @atomic_load_global_elementwise_align2(ptr addrspace(1) %ptr) {
  %val = load atomic elementwise <2 x float>, ptr addrspace(1) %ptr monotonic, align 2
  ret <2 x float> %val
}

; CHECK: error: unsupported atomic store: instruction alignment 2 is smaller than the required 8-byte alignment for this atomic operation
define void @atomic_store_global_elementwise_align2(ptr addrspace(1) %ptr, <2 x float> %val) {
  store atomic elementwise <2 x float> %val, ptr addrspace(1) %ptr monotonic, align 2
  ret void
}

; CHECK: error: unsupported atomic load: instruction alignment 4 is smaller than the required 8-byte alignment for this atomic operation
define <2 x float> @atomic_load_global_align4(ptr addrspace(1) %ptr) {
  %val = load atomic <2 x float>, ptr addrspace(1) %ptr monotonic, align 4
  ret <2 x float> %val
}

; CHECK: error: unsupported atomic load: instruction alignment 2 is smaller than the required 4-byte alignment for this atomic operation
define <2 x half> @atomic_load_global_elementwise_f16_align2(ptr addrspace(1) %ptr) {
  %val = load atomic elementwise <2 x half>, ptr addrspace(1) %ptr monotonic, align 2
  ret <2 x half> %val
}

; The LDS 4-byte rule assumes lowering to ds_read2_b32, which atomics never get.
; CHECK: error: unsupported atomic load: instruction alignment 4 is smaller than the required 8-byte alignment for this atomic operation
define <2 x float> @atomic_load_local_elementwise_align4(ptr addrspace(3) %ptr) {
  %val = load atomic elementwise <2 x float>, ptr addrspace(3) %ptr monotonic, align 4
  ret <2 x float> %val
}

; CHECK: error: unsupported atomic load: target supports atomics up to 8 bytes, but this atomic accesses 16 bytes
define <4 x float> @atomic_load_global_elementwise_v4f32_align4(ptr addrspace(1) %ptr) {
  %val = load atomic elementwise <4 x float>, ptr addrspace(1) %ptr monotonic, align 4
  ret <4 x float> %val
}
