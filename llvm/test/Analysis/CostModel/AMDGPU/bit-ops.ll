; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck -check-prefixes=ALL,SLOW16 %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefixes=ALL,FAST16 %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck -check-prefixes=ALL,SLOW16 %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefixes=ALL,FAST16 %s

; ALL: 'or_i32'
; ALL: estimated cost of 1 for {{.*}} or i32
define amdgpu_kernel void @or_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %vaddr, i32 %b) #0 {
  %vec = load i32, i32 addrspace(1)* %vaddr
  %or = or i32 %vec, %b
  store i32 %or, i32 addrspace(1)* %out
  ret void
}

; ALL: 'or_i64'
; ALL: estimated cost of 2 for {{.*}} or i64
define amdgpu_kernel void @or_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %vaddr, i64 %b) #0 {
  %vec = load i64, i64 addrspace(1)* %vaddr
  %or = or i64 %vec, %b
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

; ALL: 'or_v2i16'
; SLOW16: estimated cost of 2 for {{.*}} or <2 x i16>
; FAST16: estimated cost of 1 for {{.*}} or <2 x i16>
define amdgpu_kernel void @or_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %vaddr, <2 x i16> %b) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %vaddr
  %or = or <2 x i16> %vec, %b
  store <2 x i16> %or, <2 x i16> addrspace(1)* %out
  ret void
}

; ALL: 'xor_i32'
; ALL: estimated cost of 1 for {{.*}} xor i32
define amdgpu_kernel void @xor_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %vaddr, i32 %b) #0 {
  %vec = load i32, i32 addrspace(1)* %vaddr
  %or = xor i32 %vec, %b
  store i32 %or, i32 addrspace(1)* %out
  ret void
}

; ALL: 'xor_i64'
; ALL: estimated cost of 2 for {{.*}} xor i64
define amdgpu_kernel void @xor_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %vaddr, i64 %b) #0 {
  %vec = load i64, i64 addrspace(1)* %vaddr
  %or = xor i64 %vec, %b
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

; ALL: 'xor_v2i16'
; SLOW16: estimated cost of 2 for {{.*}} xor <2 x i16>
; FAST16: estimated cost of 1 for {{.*}} xor <2 x i16>
define amdgpu_kernel void @xor_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %vaddr, <2 x i16> %b) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %vaddr
  %xor = xor <2 x i16> %vec, %b
  store <2 x i16> %xor, <2 x i16> addrspace(1)* %out
  ret void
}

; ALL: 'and_i32'
; ALL: estimated cost of 1 for {{.*}} and i32
define amdgpu_kernel void @and_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %vaddr, i32 %b) #0 {
  %vec = load i32, i32 addrspace(1)* %vaddr
  %or = and i32 %vec, %b
  store i32 %or, i32 addrspace(1)* %out
  ret void
}

; ALL: 'and_i64'
; ALL: estimated cost of 2 for {{.*}} and i64
define amdgpu_kernel void @and_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %vaddr, i64 %b) #0 {
  %vec = load i64, i64 addrspace(1)* %vaddr
  %or = and i64 %vec, %b
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

; ALL: 'and_v2i16'
; SLOW16: estimated cost of 2 for {{.*}} and <2 x i16>
; FAST16: estimated cost of 1 for {{.*}} and <2 x i16>
define amdgpu_kernel void @and_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %vaddr, <2 x i16> %b) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %vaddr
  %and = and <2 x i16> %vec, %b
  store <2 x i16> %and, <2 x i16> addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
