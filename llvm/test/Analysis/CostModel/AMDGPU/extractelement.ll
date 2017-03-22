; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck %s

; CHECK: 'extractelement_v2i32'
; CHECK: estimated cost of 0 for {{.*}} extractelement <2 x i32>
define amdgpu_kernel void @extractelement_v2i32(i32 addrspace(1)* %out, <2 x i32> addrspace(1)* %vaddr) {
  %vec = load <2 x i32>, <2 x i32> addrspace(1)* %vaddr
  %elt = extractelement <2 x i32> %vec, i32 1
  store i32 %elt, i32 addrspace(1)* %out
  ret void
}

; CHECK: 'extractelement_v2f32'
; CHECK: estimated cost of 0 for {{.*}} extractelement <2 x float>
define amdgpu_kernel void @extractelement_v2f32(float addrspace(1)* %out, <2 x float> addrspace(1)* %vaddr) {
  %vec = load <2 x float>, <2 x float> addrspace(1)* %vaddr
  %elt = extractelement <2 x float> %vec, i32 1
  store float %elt, float addrspace(1)* %out
  ret void
}

; CHECK: 'extractelement_v3i32'
; CHECK: estimated cost of 0 for {{.*}} extractelement <3 x i32>
define amdgpu_kernel void @extractelement_v3i32(i32 addrspace(1)* %out, <3 x i32> addrspace(1)* %vaddr) {
  %vec = load <3 x i32>, <3 x i32> addrspace(1)* %vaddr
  %elt = extractelement <3 x i32> %vec, i32 1
  store i32 %elt, i32 addrspace(1)* %out
  ret void
}

; CHECK: 'extractelement_v4i32'
; CHECK: estimated cost of 0 for {{.*}} extractelement <4 x i32>
define amdgpu_kernel void @extractelement_v4i32(i32 addrspace(1)* %out, <4 x i32> addrspace(1)* %vaddr) {
  %vec = load <4 x i32>, <4 x i32> addrspace(1)* %vaddr
  %elt = extractelement <4 x i32> %vec, i32 1
  store i32 %elt, i32 addrspace(1)* %out
  ret void
}

; CHECK: 'extractelement_v8i32'
; CHECK: estimated cost of 0 for {{.*}} extractelement <8 x i32>
define amdgpu_kernel void @extractelement_v8i32(i32 addrspace(1)* %out, <8 x i32> addrspace(1)* %vaddr) {
  %vec = load <8 x i32>, <8 x i32> addrspace(1)* %vaddr
  %elt = extractelement <8 x i32> %vec, i32 1
  store i32 %elt, i32 addrspace(1)* %out
  ret void
}

; FIXME: Should be non-0
; CHECK: 'extractelement_v8i32_dynindex'
; CHECK: estimated cost of 2 for {{.*}} extractelement <8 x i32>
define amdgpu_kernel void @extractelement_v8i32_dynindex(i32 addrspace(1)* %out, <8 x i32> addrspace(1)* %vaddr, i32 %idx) {
  %vec = load <8 x i32>, <8 x i32> addrspace(1)* %vaddr
  %elt = extractelement <8 x i32> %vec, i32 %idx
  store i32 %elt, i32 addrspace(1)* %out
  ret void
}

; CHECK: 'extractelement_v2i64'
; CHECK: estimated cost of 0 for {{.*}} extractelement <2 x i64>
define amdgpu_kernel void @extractelement_v2i64(i64 addrspace(1)* %out, <2 x i64> addrspace(1)* %vaddr) {
  %vec = load <2 x i64>, <2 x i64> addrspace(1)* %vaddr
  %elt = extractelement <2 x i64> %vec, i64 1
  store i64 %elt, i64 addrspace(1)* %out
  ret void
}

; CHECK: 'extractelement_v3i64'
; CHECK: estimated cost of 0 for {{.*}} extractelement <3 x i64>
define amdgpu_kernel void @extractelement_v3i64(i64 addrspace(1)* %out, <3 x i64> addrspace(1)* %vaddr) {
  %vec = load <3 x i64>, <3 x i64> addrspace(1)* %vaddr
  %elt = extractelement <3 x i64> %vec, i64 1
  store i64 %elt, i64 addrspace(1)* %out
  ret void
}

; CHECK: 'extractelement_v4i64'
; CHECK: estimated cost of 0 for {{.*}} extractelement <4 x i64>
define amdgpu_kernel void @extractelement_v4i64(i64 addrspace(1)* %out, <4 x i64> addrspace(1)* %vaddr) {
  %vec = load <4 x i64>, <4 x i64> addrspace(1)* %vaddr
  %elt = extractelement <4 x i64> %vec, i64 1
  store i64 %elt, i64 addrspace(1)* %out
  ret void
}

; CHECK: 'extractelement_v8i64'
; CHECK: estimated cost of 0 for {{.*}} extractelement <8 x i64>
define amdgpu_kernel void @extractelement_v8i64(i64 addrspace(1)* %out, <8 x i64> addrspace(1)* %vaddr) {
  %vec = load <8 x i64>, <8 x i64> addrspace(1)* %vaddr
  %elt = extractelement <8 x i64> %vec, i64 1
  store i64 %elt, i64 addrspace(1)* %out
  ret void
}

; CHECK: 'extractelement_v4i8'
; CHECK: estimated cost of 0 for {{.*}} extractelement <4 x i8>
define amdgpu_kernel void @extractelement_v4i8(i8 addrspace(1)* %out, <4 x i8> addrspace(1)* %vaddr) {
  %vec = load <4 x i8>, <4 x i8> addrspace(1)* %vaddr
  %elt = extractelement <4 x i8> %vec, i8 1
  store i8 %elt, i8 addrspace(1)* %out
  ret void
}

; CHECK: 'extractelement_v2i16'
; CHECK: estimated cost of 0 for {{.*}} extractelement <2 x i16>
define amdgpu_kernel void @extractelement_v2i16(i16 addrspace(1)* %out, <2 x i16> addrspace(1)* %vaddr) {
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %vaddr
  %elt = extractelement <2 x i16> %vec, i16 1
  store i16 %elt, i16 addrspace(1)* %out
  ret void
}
