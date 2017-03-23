; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck %s

; CHECK: 'mul_i32'
; CHECK: estimated cost of 3 for {{.*}} mul i32
define amdgpu_kernel void @mul_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %vaddr, i32 %b) #0 {
  %vec = load i32, i32 addrspace(1)* %vaddr
  %mul = mul i32 %vec, %b
  store i32 %mul, i32 addrspace(1)* %out
  ret void
}

; CHECK: 'mul_v2i32'
; CHECK: estimated cost of 6 for {{.*}} mul <2 x i32>
define amdgpu_kernel void @mul_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %vaddr, <2 x i32> %b) #0 {
  %vec = load <2 x i32>, <2 x i32> addrspace(1)* %vaddr
  %mul = mul <2 x i32> %vec, %b
  store <2 x i32> %mul, <2 x i32> addrspace(1)* %out
  ret void
}

; CHECK: 'mul_v3i32'
; CHECK: estimated cost of 9 for {{.*}} mul <3 x i32>
define amdgpu_kernel void @mul_v3i32(<3 x i32> addrspace(1)* %out, <3 x i32> addrspace(1)* %vaddr, <3 x i32> %b) #0 {
  %vec = load <3 x i32>, <3 x i32> addrspace(1)* %vaddr
  %mul = mul <3 x i32> %vec, %b
  store <3 x i32> %mul, <3 x i32> addrspace(1)* %out
  ret void
}

; CHECK: 'mul_v4i32'
; CHECK: estimated cost of 12 for {{.*}} mul <4 x i32>
define amdgpu_kernel void @mul_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %vaddr, <4 x i32> %b) #0 {
  %vec = load <4 x i32>, <4 x i32> addrspace(1)* %vaddr
  %mul = mul <4 x i32> %vec, %b
  store <4 x i32> %mul, <4 x i32> addrspace(1)* %out
  ret void
}

; CHECK: 'mul_i64'
; CHECK: estimated cost of 16 for {{.*}} mul i64
define amdgpu_kernel void @mul_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %vaddr, i64 %b) #0 {
  %vec = load i64, i64 addrspace(1)* %vaddr
  %mul = mul i64 %vec, %b
  store i64 %mul, i64 addrspace(1)* %out
  ret void
}

; CHECK: 'mul_v2i64'
; CHECK: estimated cost of 32 for {{.*}} mul <2 x i64>
define amdgpu_kernel void @mul_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* %vaddr, <2 x i64> %b) #0 {
  %vec = load <2 x i64>, <2 x i64> addrspace(1)* %vaddr
  %mul = mul <2 x i64> %vec, %b
  store <2 x i64> %mul, <2 x i64> addrspace(1)* %out
  ret void
}

; CHECK: 'mul_v3i64'
; CHECK: estimated cost of 48 for {{.*}} mul <3 x i64>
define amdgpu_kernel void @mul_v3i64(<3 x i64> addrspace(1)* %out, <3 x i64> addrspace(1)* %vaddr, <3 x i64> %b) #0 {
  %vec = load <3 x i64>, <3 x i64> addrspace(1)* %vaddr
  %mul = mul <3 x i64> %vec, %b
  store <3 x i64> %mul, <3 x i64> addrspace(1)* %out
  ret void
}

; CHECK: 'mul_v4i64'
; CHECK: estimated cost of 64 for {{.*}} mul <4 x i64>
define amdgpu_kernel void @mul_v4i64(<4 x i64> addrspace(1)* %out, <4 x i64> addrspace(1)* %vaddr, <4 x i64> %b) #0 {
  %vec = load <4 x i64>, <4 x i64> addrspace(1)* %vaddr
  %mul = mul <4 x i64> %vec, %b
  store <4 x i64> %mul, <4 x i64> addrspace(1)* %out
  ret void
}


; CHECK: 'mul_v8i64'
; CHECK: estimated cost of 128 for {{.*}} mul <8 x i64>
define amdgpu_kernel void @mul_v8i64(<8 x i64> addrspace(1)* %out, <8 x i64> addrspace(1)* %vaddr, <8 x i64> %b) #0 {
  %vec = load <8 x i64>, <8 x i64> addrspace(1)* %vaddr
  %mul = mul <8 x i64> %vec, %b
  store <8 x i64> %mul, <8 x i64> addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
