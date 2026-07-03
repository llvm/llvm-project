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
