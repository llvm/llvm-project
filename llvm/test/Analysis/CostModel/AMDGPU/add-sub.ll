; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=FAST16,ALL %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=SLOW16,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=FAST16,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=SLOW16,ALL %s


; ALL: 'add_i32'
; ALL: estimated cost of 1 for {{.*}} add i32
define amdgpu_kernel void @add_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %vaddr, i32 %b) #0 {
  %vec = load i32, i32 addrspace(1)* %vaddr
  %add = add i32 %vec, %b
  store i32 %add, i32 addrspace(1)* %out
  ret void
}

; ALL: 'add_v2i32'
; ALL: estimated cost of 2 for {{.*}} add <2 x i32>
define amdgpu_kernel void @add_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %vaddr, <2 x i32> %b) #0 {
  %vec = load <2 x i32>, <2 x i32> addrspace(1)* %vaddr
  %add = add <2 x i32> %vec, %b
  store <2 x i32> %add, <2 x i32> addrspace(1)* %out
  ret void
}

; ALL: 'add_v3i32'
; Allow for 4 when v3i32 is illegal and TargetLowering thinks it needs widening,
; and 3 when it is legal.
; ALL: estimated cost of {{[34]}} for {{.*}} add <3 x i32>
define amdgpu_kernel void @add_v3i32(<3 x i32> addrspace(1)* %out, <3 x i32> addrspace(1)* %vaddr, <3 x i32> %b) #0 {
  %vec = load <3 x i32>, <3 x i32> addrspace(1)* %vaddr
  %add = add <3 x i32> %vec, %b
  store <3 x i32> %add, <3 x i32> addrspace(1)* %out
  ret void
}

; ALL: 'add_v4i32'
; ALL: estimated cost of 4 for {{.*}} add <4 x i32>
define amdgpu_kernel void @add_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %vaddr, <4 x i32> %b) #0 {
  %vec = load <4 x i32>, <4 x i32> addrspace(1)* %vaddr
  %add = add <4 x i32> %vec, %b
  store <4 x i32> %add, <4 x i32> addrspace(1)* %out
  ret void
}

; ALL: 'add_v5i32'
; Allow for 8 when v3i32 is illegal and TargetLowering thinks it needs widening,
; and 5 when it is legal.
; ALL: estimated cost of {{[58]}} for {{.*}} add <5 x i32>
define amdgpu_kernel void @add_v5i32(<5 x i32> addrspace(1)* %out, <5 x i32> addrspace(1)* %vaddr, <5 x i32> %b) #0 {
  %vec = load <5 x i32>, <5 x i32> addrspace(1)* %vaddr
  %add = add <5 x i32> %vec, %b
  store <5 x i32> %add, <5 x i32> addrspace(1)* %out
  ret void
}

; ALL: 'add_i64'
; ALL: estimated cost of 2 for {{.*}} add i64
define amdgpu_kernel void @add_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %vaddr, i64 %b) #0 {
  %vec = load i64, i64 addrspace(1)* %vaddr
  %add = add i64 %vec, %b
  store i64 %add, i64 addrspace(1)* %out
  ret void
}

; ALL: 'add_v2i64'
; ALL: estimated cost of 4 for {{.*}} add <2 x i64>
define amdgpu_kernel void @add_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* %vaddr, <2 x i64> %b) #0 {
  %vec = load <2 x i64>, <2 x i64> addrspace(1)* %vaddr
  %add = add <2 x i64> %vec, %b
  store <2 x i64> %add, <2 x i64> addrspace(1)* %out
  ret void
}

; ALL: 'add_v3i64'
; ALL: estimated cost of 6 for {{.*}} add <3 x i64>
define amdgpu_kernel void @add_v3i64(<3 x i64> addrspace(1)* %out, <3 x i64> addrspace(1)* %vaddr, <3 x i64> %b) #0 {
  %vec = load <3 x i64>, <3 x i64> addrspace(1)* %vaddr
  %add = add <3 x i64> %vec, %b
  store <3 x i64> %add, <3 x i64> addrspace(1)* %out
  ret void
}

; ALL: 'add_v4i64'
; ALL: estimated cost of 8 for {{.*}} add <4 x i64>
define amdgpu_kernel void @add_v4i64(<4 x i64> addrspace(1)* %out, <4 x i64> addrspace(1)* %vaddr, <4 x i64> %b) #0 {
  %vec = load <4 x i64>, <4 x i64> addrspace(1)* %vaddr
  %add = add <4 x i64> %vec, %b
  store <4 x i64> %add, <4 x i64> addrspace(1)* %out
  ret void
}

; ALL: 'add_v16i64'
; ALL: estimated cost of 32 for {{.*}} add <16 x i64>
define amdgpu_kernel void @add_v16i64(<16 x i64> addrspace(1)* %out, <16 x i64> addrspace(1)* %vaddr, <16 x i64> %b) #0 {
  %vec = load <16 x i64>, <16 x i64> addrspace(1)* %vaddr
  %add = add <16 x i64> %vec, %b
  store <16 x i64> %add, <16 x i64> addrspace(1)* %out
  ret void
}

; ALL: 'add_i16'
; ALL: estimated cost of 1 for {{.*}} add i16
define amdgpu_kernel void @add_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %vaddr, i16 %b) #0 {
  %vec = load i16, i16 addrspace(1)* %vaddr
  %add = add i16 %vec, %b
  store i16 %add, i16 addrspace(1)* %out
  ret void
}

; ALL: 'add_v2i16'
; SLOW16: estimated cost of 2 for {{.*}} add <2 x i16>
; FAST16: estimated cost of 1 for {{.*}} add <2 x i16>
define amdgpu_kernel void @add_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %vaddr, <2 x i16> %b) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %vaddr
  %add = add <2 x i16> %vec, %b
  store <2 x i16> %add, <2 x i16> addrspace(1)* %out
  ret void
}

; ALL: 'sub_i32'
; ALL: estimated cost of 1 for {{.*}} sub i32
define amdgpu_kernel void @sub_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %vaddr, i32 %b) #0 {
  %vec = load i32, i32 addrspace(1)* %vaddr
  %sub = sub i32 %vec, %b
  store i32 %sub, i32 addrspace(1)* %out
  ret void
}

; ALL: 'sub_i64'
; ALL: estimated cost of 2 for {{.*}} sub i64
define amdgpu_kernel void @sub_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %vaddr, i64 %b) #0 {
  %vec = load i64, i64 addrspace(1)* %vaddr
  %sub = sub i64 %vec, %b
  store i64 %sub, i64 addrspace(1)* %out
  ret void
}
; ALL: 'sub_i16'
; ALL: estimated cost of 1 for {{.*}} sub i16
define amdgpu_kernel void @sub_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %vaddr, i16 %b) #0 {
  %vec = load i16, i16 addrspace(1)* %vaddr
  %sub = sub i16 %vec, %b
  store i16 %sub, i16 addrspace(1)* %out
  ret void
}

; ALL: 'sub_v2i16'
; SLOW16: estimated cost of 2 for {{.*}} sub <2 x i16>
; FAST16: estimated cost of 1 for {{.*}} sub <2 x i16>
define amdgpu_kernel void @sub_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %vaddr, <2 x i16> %b) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %vaddr
  %sub = sub <2 x i16> %vec, %b
  store <2 x i16> %sub, <2 x i16> addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
