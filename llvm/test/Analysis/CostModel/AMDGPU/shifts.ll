; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=ALL,FAST64,FAST16 %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=ALL,SLOW64,SLOW16 %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=ALL,SIZEALL,FAST16 %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=ALL,SIZEALL,SLOW16 %s

; ALL-LABEL: 'shl_i32'
; ALL: estimated cost of 1 for {{.*}} shl i32
define amdgpu_kernel void @shl_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %vaddr, i32 %b) #0 {
  %vec = load i32, i32 addrspace(1)* %vaddr
  %or = shl i32 %vec, %b
  store i32 %or, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'shl_i64'
; FAST64: estimated cost of 2 for {{.*}} shl i64
; SLOW64: estimated cost of 4 for {{.*}} shl i64
; SIZEALL: estimated cost of 2 for {{.*}} shl i64
define amdgpu_kernel void @shl_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %vaddr, i64 %b) #0 {
  %vec = load i64, i64 addrspace(1)* %vaddr
  %or = shl i64 %vec, %b
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'shl_i16'
; ALL: estimated cost of 1 for {{.*}} shl i16
define amdgpu_kernel void @shl_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %vaddr, i16 %b) #0 {
  %vec = load i16, i16 addrspace(1)* %vaddr
  %or = shl i16 %vec, %b
  store i16 %or, i16 addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'shl_v2i16'
; SLOW16: estimated cost of 2 for {{.*}} shl <2 x i16>
; FAST16: estimated cost of 1 for {{.*}} shl <2 x i16>
define amdgpu_kernel void @shl_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %vaddr, <2 x i16> %b) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %vaddr
  %or = shl <2 x i16> %vec, %b
  store <2 x i16> %or, <2 x i16> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'lshr_i32'
; ALL: estimated cost of 1 for {{.*}} lshr i32
define amdgpu_kernel void @lshr_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %vaddr, i32 %b) #0 {
  %vec = load i32, i32 addrspace(1)* %vaddr
  %or = lshr i32 %vec, %b
  store i32 %or, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'lshr_i64'
; FAST64: estimated cost of 2 for {{.*}} lshr i64
; SLOW64: estimated cost of 4 for {{.*}} lshr i64
; SIZEALL: estimated cost of 2 for {{.*}} lshr i64
define amdgpu_kernel void @lshr_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %vaddr, i64 %b) #0 {
  %vec = load i64, i64 addrspace(1)* %vaddr
  %or = lshr i64 %vec, %b
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'lshr_i16'
; ALL: estimated cost of 1 for {{.*}} lshr i16
define amdgpu_kernel void @lshr_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %vaddr, i16 %b) #0 {
  %vec = load i16, i16 addrspace(1)* %vaddr
  %or = lshr i16 %vec, %b
  store i16 %or, i16 addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'lshr_v2i16'
; SLOW16: estimated cost of 2 for {{.*}} lshr <2 x i16>
; FAST16: estimated cost of 1 for {{.*}} lshr <2 x i16>
define amdgpu_kernel void @lshr_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %vaddr, <2 x i16> %b) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %vaddr
  %or = lshr <2 x i16> %vec, %b
  store <2 x i16> %or, <2 x i16> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'ashr_i32'
; ALL: estimated cost of 1 for {{.*}} ashr i32
define amdgpu_kernel void @ashr_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %vaddr, i32 %b) #0 {
  %vec = load i32, i32 addrspace(1)* %vaddr
  %or = ashr i32 %vec, %b
  store i32 %or, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'ashr_i64'
; FAST64: estimated cost of 2 for {{.*}} ashr i64
; SLOW64: estimated cost of 4 for {{.*}} ashr i64
define amdgpu_kernel void @ashr_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %vaddr, i64 %b) #0 {
  %vec = load i64, i64 addrspace(1)* %vaddr
  %or = ashr i64 %vec, %b
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'ashr_i16'
; ALL: estimated cost of 1 for {{.*}} ashr i16
define amdgpu_kernel void @ashr_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %vaddr, i16 %b) #0 {
  %vec = load i16, i16 addrspace(1)* %vaddr
  %or = ashr i16 %vec, %b
  store i16 %or, i16 addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'ashr_v2i16'
; SLOW16: estimated cost of 2 for {{.*}} ashr <2 x i16>
; FAST16: estimated cost of 1 for {{.*}} ashr <2 x i16>
define amdgpu_kernel void @ashr_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %vaddr, <2 x i16> %b) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %vaddr
  %or = ashr <2 x i16> %vec, %b
  store <2 x i16> %or, <2 x i16> addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
