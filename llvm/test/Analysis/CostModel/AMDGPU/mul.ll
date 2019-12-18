; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck -check-prefixes=SLOW16,ALL %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefixes=FAST16,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck -check-prefixes=SLOW16,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefixes=FAST16,ALL %s

; ALL: 'mul_i32'
; ALL: estimated cost of 3 for {{.*}} mul i32
define amdgpu_kernel void @mul_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %vaddr, i32 %b) #0 {
  %vec = load i32, i32 addrspace(1)* %vaddr
  %mul = mul i32 %vec, %b
  store i32 %mul, i32 addrspace(1)* %out
  ret void
}

; ALL: 'mul_v2i32'
; ALL: estimated cost of 6 for {{.*}} mul <2 x i32>
define amdgpu_kernel void @mul_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %vaddr, <2 x i32> %b) #0 {
  %vec = load <2 x i32>, <2 x i32> addrspace(1)* %vaddr
  %mul = mul <2 x i32> %vec, %b
  store <2 x i32> %mul, <2 x i32> addrspace(1)* %out
  ret void
}

; ALL: 'mul_v3i32'
; Allow for 12 when v3i32 is illegal and TargetLowering thinks it needs widening,
; and 9 when it is legal.
; ALL: estimated cost of {{9|12}} for {{.*}} mul <3 x i32>
define amdgpu_kernel void @mul_v3i32(<3 x i32> addrspace(1)* %out, <3 x i32> addrspace(1)* %vaddr, <3 x i32> %b) #0 {
  %vec = load <3 x i32>, <3 x i32> addrspace(1)* %vaddr
  %mul = mul <3 x i32> %vec, %b
  store <3 x i32> %mul, <3 x i32> addrspace(1)* %out
  ret void
}

; ALL: 'mul_v5i32'
; Allow for 24 when v5i32 is illegal and TargetLowering thinks it needs widening,
; and 15 when it is legal.
; ALL: estimated cost of {{15|24}} for {{.*}} mul <5 x i32>
define amdgpu_kernel void @mul_v5i32(<5 x i32> addrspace(1)* %out, <5 x i32> addrspace(1)* %vaddr, <5 x i32> %b) #0 {
  %vec = load <5 x i32>, <5 x i32> addrspace(1)* %vaddr
  %mul = mul <5 x i32> %vec, %b
  store <5 x i32> %mul, <5 x i32> addrspace(1)* %out
  ret void
}

; ALL: 'mul_v4i32'
; ALL: estimated cost of 12 for {{.*}} mul <4 x i32>
define amdgpu_kernel void @mul_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %vaddr, <4 x i32> %b) #0 {
  %vec = load <4 x i32>, <4 x i32> addrspace(1)* %vaddr
  %mul = mul <4 x i32> %vec, %b
  store <4 x i32> %mul, <4 x i32> addrspace(1)* %out
  ret void
}

; ALL: 'mul_i64'
; ALL: estimated cost of 16 for {{.*}} mul i64
define amdgpu_kernel void @mul_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %vaddr, i64 %b) #0 {
  %vec = load i64, i64 addrspace(1)* %vaddr
  %mul = mul i64 %vec, %b
  store i64 %mul, i64 addrspace(1)* %out
  ret void
}

; ALL: 'mul_v2i64'
; ALL: estimated cost of 32 for {{.*}} mul <2 x i64>
define amdgpu_kernel void @mul_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* %vaddr, <2 x i64> %b) #0 {
  %vec = load <2 x i64>, <2 x i64> addrspace(1)* %vaddr
  %mul = mul <2 x i64> %vec, %b
  store <2 x i64> %mul, <2 x i64> addrspace(1)* %out
  ret void
}

; ALL: 'mul_v3i64'
; ALL: estimated cost of 48 for {{.*}} mul <3 x i64>
define amdgpu_kernel void @mul_v3i64(<3 x i64> addrspace(1)* %out, <3 x i64> addrspace(1)* %vaddr, <3 x i64> %b) #0 {
  %vec = load <3 x i64>, <3 x i64> addrspace(1)* %vaddr
  %mul = mul <3 x i64> %vec, %b
  store <3 x i64> %mul, <3 x i64> addrspace(1)* %out
  ret void
}

; ALL: 'mul_v4i64'
; ALL: estimated cost of 64 for {{.*}} mul <4 x i64>
define amdgpu_kernel void @mul_v4i64(<4 x i64> addrspace(1)* %out, <4 x i64> addrspace(1)* %vaddr, <4 x i64> %b) #0 {
  %vec = load <4 x i64>, <4 x i64> addrspace(1)* %vaddr
  %mul = mul <4 x i64> %vec, %b
  store <4 x i64> %mul, <4 x i64> addrspace(1)* %out
  ret void
}


; ALL: 'mul_v8i64'
; ALL: estimated cost of 128 for {{.*}} mul <8 x i64>
define amdgpu_kernel void @mul_v8i64(<8 x i64> addrspace(1)* %out, <8 x i64> addrspace(1)* %vaddr, <8 x i64> %b) #0 {
  %vec = load <8 x i64>, <8 x i64> addrspace(1)* %vaddr
  %mul = mul <8 x i64> %vec, %b
  store <8 x i64> %mul, <8 x i64> addrspace(1)* %out
  ret void
}

; ALL: 'mul_i16'
; ALL: estimated cost of 3 for {{.*}} mul i16
define amdgpu_kernel void @mul_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %vaddr, i16 %b) #0 {
  %vec = load i16, i16 addrspace(1)* %vaddr
  %mul = mul i16 %vec, %b
  store i16 %mul, i16 addrspace(1)* %out
  ret void
}

; ALL: 'mul_v2i16'
; SLOW16: estimated cost of 6 for {{.*}} mul <2 x i16>
; FAST16: estimated cost of 3 for {{.*}} mul <2 x i16>
define amdgpu_kernel void @mul_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %vaddr, <2 x i16> %b) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %vaddr
  %mul = mul <2 x i16> %vec, %b
  store <2 x i16> %mul, <2 x i16> addrspace(1)* %out
  ret void
}

; ALL: 'mul_v3i16'
; SLOW16: estimated cost of 12 for {{.*}} mul <3 x i16>
; FAST16: estimated cost of 6 for {{.*}} mul <3 x i16>
define amdgpu_kernel void @mul_v3i16(<3 x i16> addrspace(1)* %out, <3 x i16> addrspace(1)* %vaddr, <3 x i16> %b) #0 {
  %vec = load <3 x i16>, <3 x i16> addrspace(1)* %vaddr
  %mul = mul <3 x i16> %vec, %b
  store <3 x i16> %mul, <3 x i16> addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
