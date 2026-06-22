; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx950 < %s | FileCheck -check-prefixes=GCN %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx950 < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck -check-prefixes=GCN,SDAG %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck -check-prefix=GCN %s

declare i32 @llvm.amdgcn.prng.b32(i32) #0

; GCN-LABEL: {{^}}prng_b32:
; GCN: v_prng_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}
define amdgpu_kernel void @prng_b32(ptr addrspace(1) %out, i32 %src) #1 {
  %prng = call i32 @llvm.amdgcn.prng.b32(i32 %src) #0
  store i32 %prng, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}prng_b32_constant_4
; GCN: v_prng_b32_e32 {{v[0-9]+}}, 4
define amdgpu_kernel void @prng_b32_constant_4(ptr addrspace(1) %out) #1 {
  %prng = call i32 @llvm.amdgcn.prng.b32(i32 4) #0
  store i32 %prng, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}prng_b32_constant_100
; GCN: v_prng_b32_e32 {{v[0-9]+}}, 0x64
define amdgpu_kernel void @prng_b32_constant_100(ptr addrspace(1) %out) #1 {
  %prng = call i32 @llvm.amdgcn.prng.b32(i32 100) #0
  store i32 %prng, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}prng_undef_i32:
; SDAG-NOT: v_prng_b32
define amdgpu_kernel void @prng_undef_i32(ptr addrspace(1) %out) #1 {
  %prng = call i32 @llvm.amdgcn.prng.b32(i32 undef)
  store i32 %prng, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
