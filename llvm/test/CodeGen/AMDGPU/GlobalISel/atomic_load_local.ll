; RUN: llc -global-isel -new-reg-bank-select -global-isel-abort=0 -mtriple=amdgcn-amd-amdhsa -mcpu=kaveri < %s | FileCheck -check-prefixes=GCN,CI %s
; RUN: llc -global-isel -new-reg-bank-select -global-isel-abort=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefixes=GCN,GFX9 %s

; GCN-LABEL: {{^}}atomic_load_monotonic_i8:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_read_u8 v0, v0{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define i8 @atomic_load_monotonic_i8(ptr addrspace(3) %ptr) {
  %load = load atomic i8, ptr addrspace(3) %ptr monotonic, align 1
  ret i8 %load
}

; GCN-LABEL: {{^}}atomic_load_monotonic_i8_offset:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_read_u8 v0, v0 offset:16{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define i8 @atomic_load_monotonic_i8_offset(ptr addrspace(3) %ptr) {
  %gep = getelementptr inbounds i8, ptr addrspace(3) %ptr, i8 16
  %load = load atomic i8, ptr addrspace(3) %gep monotonic, align 1
  ret i8 %load
}

; GCN-LABEL: {{^}}atomic_load_monotonic_i16:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_read_u16 v0, v0{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define i16 @atomic_load_monotonic_i16(ptr addrspace(3) %ptr) {
  %load = load atomic i16, ptr addrspace(3) %ptr monotonic, align 2
  ret i16 %load
}

; GCN-LABEL: {{^}}atomic_load_monotonic_i16_offset:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_read_u16 v0, v0 offset:32{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define i16 @atomic_load_monotonic_i16_offset(ptr addrspace(3) %ptr) {
  %gep = getelementptr inbounds i16, ptr addrspace(3) %ptr, i16 16
  %load = load atomic i16, ptr addrspace(3) %gep monotonic, align 2
  ret i16 %load
}

; GCN-LABEL: {{^}}atomic_load_monotonic_i32:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_read_b32 v0, v0{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define i32 @atomic_load_monotonic_i32(ptr addrspace(3) %ptr) {
  %load = load atomic i32, ptr addrspace(3) %ptr monotonic, align 4
  ret i32 %load
}

; GCN-LABEL: {{^}}atomic_load_monotonic_i32_offset:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_read_b32 v0, v0 offset:64{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define i32 @atomic_load_monotonic_i32_offset(ptr addrspace(3) %ptr) {
  %gep = getelementptr inbounds i32, ptr addrspace(3) %ptr, i32 16
  %load = load atomic i32, ptr addrspace(3) %gep monotonic, align 4
  ret i32 %load
}

; GCN-LABEL: {{^}}atomic_load_monotonic_i64:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_read_b64 v[0:1], v0{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define i64 @atomic_load_monotonic_i64(ptr addrspace(3) %ptr) {
  %load = load atomic i64, ptr addrspace(3) %ptr monotonic, align 8
  ret i64 %load
}

; GCN-LABEL: {{^}}atomic_load_monotonic_i64_offset:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_read_b64 v[0:1], v0 offset:128{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define i64 @atomic_load_monotonic_i64_offset(ptr addrspace(3) %ptr) {
  %gep = getelementptr inbounds i64, ptr addrspace(3) %ptr, i32 16
  %load = load atomic i64, ptr addrspace(3) %gep monotonic, align 8
  ret i64 %load
}

; GCN-LABEL: {{^}}atomic_load_monotonic_f32_offset:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_read_b32 v0, v0 offset:64{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define float @atomic_load_monotonic_f32_offset(ptr addrspace(3) %ptr) {
  %gep = getelementptr inbounds float, ptr addrspace(3) %ptr, i32 16
  %load = load atomic float, ptr addrspace(3) %gep monotonic, align 4
  ret float %load
}

; GCN-LABEL: {{^}}atomic_load_monotonic_f64_offset:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_read_b64 v[0:1], v0 offset:128{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define double @atomic_load_monotonic_f64_offset(ptr addrspace(3) %ptr) {
  %gep = getelementptr inbounds double, ptr addrspace(3) %ptr, i32 16
  %load = load atomic double, ptr addrspace(3) %gep monotonic, align 8
  ret double %load
}

; GCN-LABEL: {{^}}atomic_load_monotonic_p0i8_offset:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_read_b64 v[0:1], v0 offset:128{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define ptr @atomic_load_monotonic_p0i8_offset(ptr addrspace(3) %ptr) {
  %gep = getelementptr inbounds ptr, ptr addrspace(3) %ptr, i32 16
  %load = load atomic ptr, ptr addrspace(3) %gep monotonic, align 8
  ret ptr %load
}

; GCN-LABEL: {{^}}atomic_load_monotonic_p3i8_offset:
; GCN: s_waitcnt
; GFX9-NOT: s_mov_b32 m0
; CI-NEXT: s_mov_b32 m0
; GCN-NEXT: ds_read_b32 v0, v0 offset:64{{$}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define ptr addrspace(3) @atomic_load_monotonic_p3i8_offset(ptr addrspace(3) %ptr) {
  %gep = getelementptr inbounds ptr addrspace(3), ptr addrspace(3) %ptr, i32 16
  %load = load atomic ptr addrspace(3), ptr addrspace(3) %gep monotonic, align 4
  ret ptr addrspace(3) %load
}
