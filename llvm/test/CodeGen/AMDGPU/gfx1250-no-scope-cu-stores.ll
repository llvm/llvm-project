; RUN: llc -mtriple=amdgcn-amd-amdhsa -O3 -mcpu=gfx1250 < %s | FileCheck --check-prefixes=GCN,CU %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -O3 -mcpu=gfx1250 -mattr=-cu-stores < %s | FileCheck --check-prefixes=GCN,NOCU %s

; Check that if -cu-stores is used, we use SCOPE_SE minimum on all stores.

; GCN:     flat_store:
; CU:        flat_store_b32 v{{.*}}, v{{.*}}, s{{.*}} scope:SCOPE_SE
; NOCU:      flat_store_b32 v{{.*}}, v{{.*}}, s{{.*}} scope:SCOPE_SE
; GCN:     .amdhsa_kernel flat_store
; CU:       .amdhsa_uses_cu_stores 1
; NOCU:     .amdhsa_uses_cu_stores 0
define amdgpu_kernel void @flat_store(ptr %dst, i32 %val) {
entry:
  store i32 %val, ptr %dst
  ret void
}

; GCN:     global_store:
; CU:        global_store_b32 v{{.*}}, v{{.*}}, s{{.*}}{{$}}
; NOCU:      global_store_b32 v{{.*}}, v{{.*}}, s{{.*}} scope:SCOPE_SE
; GCN:     .amdhsa_kernel global_store
; CU:        .amdhsa_uses_cu_stores 1
; NOCU:      .amdhsa_uses_cu_stores 0
define amdgpu_kernel void @global_store(ptr addrspace(1) %dst, i32 %val) {
entry:
  store i32 %val, ptr addrspace(1) %dst
  ret void
}

; GCN:     local_store:
; CU:        ds_store_b32 v{{.*}}, v{{.*}}{{$}}
; NOCU:      ds_store_b32 v{{.*}}, v{{.*}}{{$}}
; GCN:     .amdhsa_kernel local_store
; CU:        .amdhsa_uses_cu_stores 1
; NOCU:     .amdhsa_uses_cu_stores 0
define amdgpu_kernel void @local_store(ptr addrspace(3) %dst, i32 %val) {
entry:
  store i32 %val, ptr addrspace(3) %dst
  ret void
}

; GCN:     scratch_store:
; CU:        scratch_store_b32 off, v{{.*}}, s{{.*}} scope:SCOPE_SE
; NOCU:      scratch_store_b32 off, v{{.*}}, s{{.*}} scope:SCOPE_SE
; GCN:     .amdhsa_kernel scratch_store
; CU:        .amdhsa_uses_cu_stores 1
; NOCU:      .amdhsa_uses_cu_stores 0
define amdgpu_kernel void @scratch_store(ptr addrspace(5) %dst, i32 %val) {
entry:
  store i32 %val, ptr addrspace(5) %dst
  ret void
}

; GCN:     flat_atomic_store:
; CU:        flat_store_b32 v{{.*}}, v{{.*}}, s{{.*}} scope:SCOPE_SE
; NOCU:      flat_store_b32 v{{.*}}, v{{.*}}, s{{.*}} scope:SCOPE_SE
; GCN:     .amdhsa_kernel flat_atomic_store
; CU:        .amdhsa_uses_cu_stores 1
; NOCU:      .amdhsa_uses_cu_stores 0
define amdgpu_kernel void @flat_atomic_store(ptr %dst, i32 %val) {
entry:
  store atomic i32 %val, ptr %dst syncscope("wavefront") unordered, align 4
  ret void
}

; GCN:     global_atomic_store:
; CU:        global_store_b32 v{{.*}}, v{{.*}}, s{{.*}}{{$}}
; NOCU:      global_store_b32  v{{.*}}, v{{.*}}, s{{.*}} scope:SCOPE_SE
; GCN:     .amdhsa_kernel global_atomic_store
; CU:        .amdhsa_uses_cu_stores 1
; NOCU:      .amdhsa_uses_cu_stores 0
define amdgpu_kernel void @global_atomic_store(ptr addrspace(1) %dst, i32 %val) {
entry:
  store atomic i32 %val, ptr addrspace(1) %dst syncscope("wavefront") unordered, align 4
  ret void
}

; GCN:     local_atomic_store:
; CU:        ds_store_b32 v{{.*}}, v{{.*}}{{$}}
; NOCU:      ds_store_b32 v{{.*}}, v{{.*}}{{$}}
; GCN:     .amdhsa_kernel local_atomic_store
; CU:        .amdhsa_uses_cu_stores 1
; NOCU:      .amdhsa_uses_cu_stores 0
define amdgpu_kernel void @local_atomic_store(ptr addrspace(3) %dst, i32 %val) {
entry:
  store atomic i32 %val, ptr addrspace(3) %dst syncscope("wavefront") unordered, align 4
  ret void
}
