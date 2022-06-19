; RUN: llc -global-isel -mtriple=amdgcn-amd-amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,CI %s
; RUN: llc -global-isel -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9 %s

; GCN-LABEL: {{^}}atomic_store_monotonic_i8:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_write_b8 v0, v1{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @atomic_store_monotonic_i8(i8 addrspace(3)* %ptr, i8 %val) {
  store atomic i8 %val, i8 addrspace(3)* %ptr monotonic, align 1
  ret void
}

; GCN-LABEL: {{^}}atomic_store_monotonic_offset_i8:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_write_b8 v0, v1 offset:16{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @atomic_store_monotonic_offset_i8(i8 addrspace(3)* %ptr, i8 %val) {
  %gep = getelementptr inbounds i8, i8 addrspace(3)* %ptr, i8 16
  store atomic i8 %val, i8 addrspace(3)* %gep monotonic, align 1
  ret void
}

; GCN-LABEL: {{^}}atomic_store_monotonic_i16:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_write_b16 v0, v1{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @atomic_store_monotonic_i16(i16 addrspace(3)* %ptr, i16 %val) {
  store atomic i16 %val, i16 addrspace(3)* %ptr monotonic, align 2
  ret void
}

; GCN-LABEL: {{^}}atomic_store_monotonic_offset_i16:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_write_b16 v0, v1 offset:32{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @atomic_store_monotonic_offset_i16(i16 addrspace(3)* %ptr, i16 %val) {
  %gep = getelementptr inbounds i16, i16 addrspace(3)* %ptr, i16 16
  store atomic i16 %val, i16 addrspace(3)* %gep monotonic, align 2
  ret void
}

; GCN-LABEL: {{^}}atomic_store_monotonic_i32:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_write_b32 v0, v1{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @atomic_store_monotonic_i32(i32 addrspace(3)* %ptr, i32 %val) {
  store atomic i32 %val, i32 addrspace(3)* %ptr monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_store_monotonic_offset_i32:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_write_b32 v0, v1 offset:64{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @atomic_store_monotonic_offset_i32(i32 addrspace(3)* %ptr, i32 %val) {
  %gep = getelementptr inbounds i32, i32 addrspace(3)* %ptr, i32 16
  store atomic i32 %val, i32 addrspace(3)* %gep monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_store_monotonic_i64:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_write_b64 v0, v[1:2]{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @atomic_store_monotonic_i64(i64 addrspace(3)* %ptr, i64 %val) {
  store atomic i64 %val, i64 addrspace(3)* %ptr monotonic, align 8
  ret void
}

; GCN-LABEL: {{^}}atomic_store_monotonic_offset_i64:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_write_b64 v0, v[1:2] offset:128{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @atomic_store_monotonic_offset_i64(i64 addrspace(3)* %ptr, i64 %val) {
  %gep = getelementptr inbounds i64, i64 addrspace(3)* %ptr, i64 16
  store atomic i64 %val, i64 addrspace(3)* %gep monotonic, align 8
  ret void
}

