; RUN: llc -mtriple=amdgcn -mcpu=tahiti < %s | FileCheck -check-prefixes=GCN,SICIVI,SICI,SI %s
; RUN: llc -mtriple=amdgcn -mcpu=bonaire < %s | FileCheck -check-prefixes=GCN,SICIVI,SICI %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga < %s | FileCheck -check-prefixes=GCN,SICIVI,VI %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck -check-prefixes=GCN,GFX9 %s

; GCN-LABEL: {{^}}load_i32:
; GCN-DAG: s_mov_b32 s3, 0
; GCN-DAG: s_mov_b32 s2, s1
; GCN-DAG: s_mov_b32 s1, s3
; SICI-DAG: s_load_dword s{{[0-9]}}, s[0:1], 0x0
; SICI-DAG: s_load_dword s{{[0-9]}}, s[2:3], 0x2
; GFX9-DAG: s_load_dword s{{[0-9]}}, s[0:1], 0x0
; GFX9-DAG: s_load_dword s{{[0-9]}}, s[2:3], 0x8
define amdgpu_vs float @load_i32(ptr addrspace(6) inreg %p0, ptr addrspace(6) inreg %p1) #0 {
  %gep1 = getelementptr inbounds i32, ptr addrspace(6) %p1, i32 2
  %r0 = load i32, ptr addrspace(6) %p0
  %r1 = load i32, ptr addrspace(6) %gep1
  %r = add i32 %r0, %r1
  %r2 = bitcast i32 %r to float
  ret float %r2
}

; GCN-LABEL: {{^}}load_v2i32:
; SICIVI-DAG: s_mov_b32 s3, 0
; SICIVI-DAG: s_mov_b32 s2, s1
; SICIVI-DAG: s_mov_b32 s1, s3
; SICI-DAG: s_load_dwordx2 s[{{.*}}], s[0:1], 0x0
; SICI-DAG: s_load_dwordx2 s[{{.*}}], s[2:3], 0x4
; VI-DAG: s_load_dwordx2 s[{{.*}}], s[0:1], 0x0
; VI-DAG: s_load_dwordx2 s[{{.*}}], s[2:3], 0x10
; GFX9-DAG: s_mov_b32 s2, s1
; GFX9-DAG: s_mov_b32 s3, 0
; GFX9-DAG: s_mov_b32 s1, s3
; GFX9-DAG: s_load_dwordx2 s[{{.*}}], s[0:1], 0x0
; GFX9-DAG: s_load_dwordx2 s[{{.*}}], s[2:3], 0x10
define amdgpu_vs <2 x float> @load_v2i32(ptr addrspace(6) inreg %p0, ptr addrspace(6) inreg %p1) #0 {
  %gep1 = getelementptr inbounds <2 x i32>, ptr addrspace(6) %p1, i32 2
  %r0 = load <2 x i32>, ptr addrspace(6) %p0
  %r1 = load <2 x i32>, ptr addrspace(6) %gep1
  %r = add <2 x i32> %r0, %r1
  %r2 = bitcast <2 x i32> %r to <2 x float>
  ret <2 x float> %r2
}

; GCN-LABEL: {{^}}load_v4i32:
; GCN-DAG: s_mov_b32 s3, 0
; GCN-DAG: s_mov_b32 s2, s1
; GCN-DAG: s_mov_b32 s1, s3
; SICI-DAG: s_load_dwordx4 s[{{.*}}], s[0:1], 0x0
; SICI-DAG: s_load_dwordx4 s[{{.*}}], s[2:3], 0x8
; VI-DAG: s_load_dwordx4 s[{{.*}}], s[0:1], 0x0
; VI-DAG: s_load_dwordx4 s[{{.*}}], s[2:3], 0x20
; GFX9-DAG: s_load_dwordx4 s[{{.*}}], s[0:1], 0x0
; GFX9-DAG: s_load_dwordx4 s[{{.*}}], s[2:3], 0x20
define amdgpu_vs <4 x float> @load_v4i32(ptr addrspace(6) inreg %p0, ptr addrspace(6) inreg %p1) #0 {
  %gep1 = getelementptr inbounds <4 x i32>, ptr addrspace(6) %p1, i32 2
  %r0 = load <4 x i32>, ptr addrspace(6) %p0
  %r1 = load <4 x i32>, ptr addrspace(6) %gep1
  %r = add <4 x i32> %r0, %r1
  %r2 = bitcast <4 x i32> %r to <4 x float>
  ret <4 x float> %r2
}

; GCN-LABEL: {{^}}load_v8i32:
; GCN-DAG: s_mov_b32 s3, 0
; GCN-DAG: s_mov_b32 s2, s1
; GCN-DAG: s_mov_b32 s1, s3
; SICI-DAG: s_load_dwordx8 s[{{.*}}], s[0:1], 0x0
; SICI-DAG: s_load_dwordx8 s[{{.*}}], s[2:3], 0x10
; VI-DAG: s_load_dwordx8 s[{{.*}}], s[0:1], 0x0
; VI-DAG: s_load_dwordx8 s[{{.*}}], s[2:3], 0x40
; GFX9-DAG: s_load_dwordx8 s[{{.*}}], s[0:1], 0x0
; GFX9-DAG: s_load_dwordx8 s[{{.*}}], s[2:3], 0x40
define amdgpu_vs <8 x float> @load_v8i32(ptr addrspace(6) inreg %p0, ptr addrspace(6) inreg %p1) #0 {
  %gep1 = getelementptr inbounds <8 x i32>, ptr addrspace(6) %p1, i32 2
  %r0 = load <8 x i32>, ptr addrspace(6) %p0
  %r1 = load <8 x i32>, ptr addrspace(6) %gep1
  %r = add <8 x i32> %r0, %r1
  %r2 = bitcast <8 x i32> %r to <8 x float>
  ret <8 x float> %r2
}

; GCN-LABEL: {{^}}load_v16i32:
; GCN-DAG: s_mov_b32 s3, 0
; GCN-DAG: s_mov_b32 s2, s1
; GCN-DAG: s_mov_b32 s1, s3
; SICI-DAG: s_load_dwordx16 s[{{.*}}], s[0:1], 0x0
; SICI-DAG: s_load_dwordx16 s[{{.*}}], s[2:3], 0x20
; VI-DAG: s_load_dwordx16 s[{{.*}}], s[0:1], 0x0
; VI-DAG: s_load_dwordx16 s[{{.*}}], s[2:3], 0x80
; GFX9-DAG: s_load_dwordx16 s[{{.*}}], s[0:1], 0x0
; GFX9-DAG: s_load_dwordx16 s[{{.*}}], s[2:3], 0x80
define amdgpu_vs <16 x float> @load_v16i32(ptr addrspace(6) inreg %p0, ptr addrspace(6) inreg %p1) #0 {
  %gep1 = getelementptr inbounds <16 x i32>, ptr addrspace(6) %p1, i32 2
  %r0 = load <16 x i32>, ptr addrspace(6) %p0
  %r1 = load <16 x i32>, ptr addrspace(6) %gep1
  %r = add <16 x i32> %r0, %r1
  %r2 = bitcast <16 x i32> %r to <16 x float>
  ret <16 x float> %r2
}

; GCN-LABEL: {{^}}load_float:
; GCN-DAG: s_mov_b32 s3, 0
; GCN-DAG: s_mov_b32 s2, s1
; GCN-DAG: s_mov_b32 s1, s3
; SICI-DAG: s_load_dword s{{[0-9]}}, s[0:1], 0x0
; SICI-DAG: s_load_dword s{{[0-9]}}, s[2:3], 0x2
; VI-DAG: s_load_dword s{{[0-9]}}, s[0:1], 0x0
; VI-DAG: s_load_dword s{{[0-9]}}, s[2:3], 0x8
; GFX9-DAG: s_load_dword s{{[0-9]}}, s[0:1], 0x0
; GFX9-DAG: s_load_dword s{{[0-9]}}, s[2:3], 0x8
define amdgpu_vs float @load_float(ptr addrspace(6) inreg %p0, ptr addrspace(6) inreg %p1) #0 {
  %gep1 = getelementptr inbounds float, ptr addrspace(6) %p1, i32 2
  %r0 = load float, ptr addrspace(6) %p0
  %r1 = load float, ptr addrspace(6) %gep1
  %r = fadd float %r0, %r1
  ret float %r
}

; GCN-LABEL: {{^}}load_v2float:
; SICIVI-DAG: s_mov_b32 s3, 0
; SICIVI-DAG: s_mov_b32 s2, s1
; SICIVI-DAG: s_mov_b32 s1, s3
; SICI-DAG: s_load_dwordx2 s[{{.*}}], s[0:1], 0x0
; SICI-DAG: s_load_dwordx2 s[{{.*}}], s[2:3], 0x4
; VI-DAG: s_load_dwordx2 s[{{.*}}], s[0:1], 0x0
; VI-DAG: s_load_dwordx2 s[{{.*}}], s[2:3], 0x10
; GFX9-DAG: s_mov_b32 s2, s1
; GFX9-DAG: s_mov_b32 s3, 0
; GFX9-DAG: s_mov_b32 s1, s3
; GFX9-DAG: s_load_dwordx2 s[{{.*}}], s[0:1], 0x0
; GFX9-DAG: s_load_dwordx2 s[{{.*}}], s[2:3], 0x10
define amdgpu_vs <2 x float> @load_v2float(ptr addrspace(6) inreg %p0, ptr addrspace(6) inreg %p1) #0 {
  %gep1 = getelementptr inbounds <2 x float>, ptr addrspace(6) %p1, i32 2
  %r0 = load <2 x float>, ptr addrspace(6) %p0
  %r1 = load <2 x float>, ptr addrspace(6) %gep1
  %r = fadd <2 x float> %r0, %r1
  ret <2 x float> %r
}

; GCN-LABEL: {{^}}load_v4float:
; GCN-DAG: s_mov_b32 s3, 0
; GCN-DAG: s_mov_b32 s2, s1
; GCN-DAG: s_mov_b32 s1, s3
; SICI-DAG: s_load_dwordx4 s[{{.*}}], s[0:1], 0x0
; SICI-DAG: s_load_dwordx4 s[{{.*}}], s[2:3], 0x8
; VI-DAG: s_load_dwordx4 s[{{.*}}], s[0:1], 0x0
; VI-DAG: s_load_dwordx4 s[{{.*}}], s[2:3], 0x20
; GFX9-DAG: s_load_dwordx4 s[{{.*}}], s[0:1], 0x0
; GFX9-DAG: s_load_dwordx4 s[{{.*}}], s[2:3], 0x20
define amdgpu_vs <4 x float> @load_v4float(ptr addrspace(6) inreg %p0, ptr addrspace(6) inreg %p1) #0 {
  %gep1 = getelementptr inbounds <4 x float>, ptr addrspace(6) %p1, i32 2
  %r0 = load <4 x float>, ptr addrspace(6) %p0
  %r1 = load <4 x float>, ptr addrspace(6) %gep1
  %r = fadd <4 x float> %r0, %r1
  ret <4 x float> %r
}

; GCN-LABEL: {{^}}load_v8float:
; GCN-DAG: s_mov_b32 s3, 0
; GCN-DAG: s_mov_b32 s2, s1
; GCN-DAG: s_mov_b32 s1, s3
; SICI-DAG: s_load_dwordx8 s[{{.*}}], s[0:1], 0x0
; SICI-DAG: s_load_dwordx8 s[{{.*}}], s[2:3], 0x10
; VI-DAG: s_load_dwordx8 s[{{.*}}], s[0:1], 0x0
; VI-DAG: s_load_dwordx8 s[{{.*}}], s[2:3], 0x40
; GFX9-DAG: s_load_dwordx8 s[{{.*}}], s[0:1], 0x0
; GFX9-DAG: s_load_dwordx8 s[{{.*}}], s[2:3], 0x40
define amdgpu_vs <8 x float> @load_v8float(ptr addrspace(6) inreg %p0, ptr addrspace(6) inreg %p1) #0 {
  %gep1 = getelementptr inbounds <8 x float>, ptr addrspace(6) %p1, i32 2
  %r0 = load <8 x float>, ptr addrspace(6) %p0
  %r1 = load <8 x float>, ptr addrspace(6) %gep1
  %r = fadd <8 x float> %r0, %r1
  ret <8 x float> %r
}

; GCN-LABEL: {{^}}load_v16float:
; GCN-DAG: s_mov_b32 s3, 0
; GCN-DAG: s_mov_b32 s2, s1
; GCN-DAG: s_mov_b32 s1, s3
; SICI-DAG: s_load_dwordx16 s[{{.*}}], s[0:1], 0x0
; SICI-DAG: s_load_dwordx16 s[{{.*}}], s[2:3], 0x20
; VI-DAG: s_load_dwordx16 s[{{.*}}], s[0:1], 0x0
; VI-DAG: s_load_dwordx16 s[{{.*}}], s[2:3], 0x80
; GFX9-DAG: s_load_dwordx16 s[{{.*}}], s[0:1], 0x0
; GFX9-DAG: s_load_dwordx16 s[{{.*}}], s[2:3], 0x80
define amdgpu_vs <16 x float> @load_v16float(ptr addrspace(6) inreg %p0, ptr addrspace(6) inreg %p1) #0 {
  %gep1 = getelementptr inbounds <16 x float>, ptr addrspace(6) %p1, i32 2
  %r0 = load <16 x float>, ptr addrspace(6) %p0
  %r1 = load <16 x float>, ptr addrspace(6) %gep1
  %r = fadd <16 x float> %r0, %r1
  ret <16 x float> %r
}

; GCN-LABEL: {{^}}load_i32_hi0:
; GCN: s_mov_b32 s1, 0
; GCN-NEXT: s_load_dword s0, s[0:1], 0x0
define amdgpu_vs i32 @load_i32_hi0(ptr addrspace(6) inreg %p) #1 {
  %r0 = load i32, ptr addrspace(6) %p
  ret i32 %r0
}

; GCN-LABEL: {{^}}load_i32_hi1:
; GCN: s_mov_b32 s1, 1
; GCN-NEXT: s_load_dword s0, s[0:1], 0x0
define amdgpu_vs i32 @load_i32_hi1(ptr addrspace(6) inreg %p) #2 {
  %r0 = load i32, ptr addrspace(6) %p
  ret i32 %r0
}

; GCN-LABEL: {{^}}load_i32_hiffff8000:
; GCN: s_movk_i32 s1, 0x8000
; GCN-NEXT: s_load_dword s0, s[0:1], 0x0
define amdgpu_vs i32 @load_i32_hiffff8000(ptr addrspace(6) inreg %p) #3 {
  %r0 = load i32, ptr addrspace(6) %p
  ret i32 %r0
}

; GCN-LABEL: {{^}}load_i32_hifffffff0:
; GCN: s_mov_b32 s1, -16
; GCN-NEXT: s_load_dword s0, s[0:1], 0x0
define amdgpu_vs i32 @load_i32_hifffffff0(ptr addrspace(6) inreg %p) #4 {
  %r0 = load i32, ptr addrspace(6) %p
  ret i32 %r0
}

; GCN-LABEL: {{^}}load_sampler
; GCN: v_readfirstlane_b32
; SI: s_nop
; GCN: s_load_dwordx8
; GCN-NEXT: s_load_dwordx4
; GCN: image_sample
define amdgpu_ps <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> @load_sampler(ptr addrspace(6) inreg noalias dereferenceable(18446744073709551615), ptr addrspace(6) inreg noalias dereferenceable(18446744073709551615), ptr addrspace(6) inreg noalias dereferenceable(18446744073709551615), ptr addrspace(6) inreg noalias dereferenceable(18446744073709551615), float inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, i32, i32, float, i32) #5 {
main_body:
  %22 = call nsz float @llvm.amdgcn.interp.mov(i32 2, i32 0, i32 0, i32 %5) #8
  %23 = bitcast float %22 to i32
  %24 = shl i32 %23, 1
  %25 = getelementptr inbounds [0 x <8 x i32>], ptr addrspace(6) %1, i32 0, i32 %24, !amdgpu.uniform !0
  %26 = load <8 x i32>, ptr addrspace(6) %25, align 32, !invariant.load !0
  %27 = shl i32 %23, 2
  %28 = or i32 %27, 3
  %29 = getelementptr inbounds [0 x <4 x i32>], ptr addrspace(6) %1, i32 0, i32 %28, !amdgpu.uniform !0
  %30 = load <4 x i32>, ptr addrspace(6) %29, align 16, !invariant.load !0
  %31 = call nsz <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float 0.0, <8 x i32> %26, <4 x i32> %30, i1 0, i32 0, i32 0) #8
  %32 = extractelement <4 x float> %31, i32 0
  %33 = extractelement <4 x float> %31, i32 1
  %34 = extractelement <4 x float> %31, i32 2
  %35 = extractelement <4 x float> %31, i32 3
  %36 = bitcast float %4 to i32
  %37 = insertvalue <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> undef, i32 %36, 4
  %38 = insertvalue <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %37, float %32, 5
  %39 = insertvalue <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %38, float %33, 6
  %40 = insertvalue <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %39, float %34, 7
  %41 = insertvalue <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %40, float %35, 8
  %42 = insertvalue <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %41, float %20, 19
  ret <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %42
}

; GCN-LABEL: {{^}}load_sampler_nouniform
; GCN: v_readfirstlane_b32
; SI: s_nop
; GCN: s_load_dwordx8
; GCN-NEXT: s_load_dwordx4
; GCN: image_sample
define amdgpu_ps <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> @load_sampler_nouniform(ptr addrspace(6) inreg noalias dereferenceable(18446744073709551615), ptr addrspace(6) inreg noalias dereferenceable(18446744073709551615), ptr addrspace(6) inreg noalias dereferenceable(18446744073709551615), ptr addrspace(6) inreg noalias dereferenceable(18446744073709551615), float inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, i32, i32, float, i32) #5 {
main_body:
  %22 = call nsz float @llvm.amdgcn.interp.mov(i32 2, i32 0, i32 0, i32 %5) #8
  %23 = bitcast float %22 to i32
  %24 = shl i32 %23, 1
  %25 = getelementptr inbounds [0 x <8 x i32>], ptr addrspace(6) %1, i32 0, i32 %24
  %26 = load <8 x i32>, ptr addrspace(6) %25, align 32, !invariant.load !0
  %27 = shl i32 %23, 2
  %28 = or i32 %27, 3
  %29 = getelementptr inbounds [0 x <4 x i32>], ptr addrspace(6) %1, i32 0, i32 %28
  %30 = load <4 x i32>, ptr addrspace(6) %29, align 16, !invariant.load !0
  %31 = call nsz <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float 0.0, <8 x i32> %26, <4 x i32> %30, i1 0, i32 0, i32 0) #8
  %32 = extractelement <4 x float> %31, i32 0
  %33 = extractelement <4 x float> %31, i32 1
  %34 = extractelement <4 x float> %31, i32 2
  %35 = extractelement <4 x float> %31, i32 3
  %36 = bitcast float %4 to i32
  %37 = insertvalue <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> undef, i32 %36, 4
  %38 = insertvalue <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %37, float %32, 5
  %39 = insertvalue <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %38, float %33, 6
  %40 = insertvalue <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %39, float %34, 7
  %41 = insertvalue <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %40, float %35, 8
  %42 = insertvalue <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %41, float %20, 19
  ret <{ i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> %42
}

; GCN-LABEL: {{^}}load_addr_no_fold:
; GCN-DAG: s_add_i32 s0, s0, 4
; GCN-DAG: s_mov_b32 s1, 0
; GCN: s_load_dword s{{[0-9]}}, s[0:1], 0x0
define amdgpu_vs float @load_addr_no_fold(ptr addrspace(6) inreg noalias %p0) #0 {
  %gep1 = getelementptr i32, ptr addrspace(6) %p0, i32 1
  %r1 = load i32, ptr addrspace(6) %gep1
  %r2 = bitcast i32 %r1 to float
  ret float %r2
}

; GCN-LABEL: {{^}}vgpr_arg_src:
; GCN: v_readfirstlane_b32 s[[READLANE:[0-9]+]], v0
; GCN: s_mov_b32 s[[ZERO:[0-9]+]]
; GCN: s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s[[[READLANE]]:[[ZERO]]]
define amdgpu_vs float @vgpr_arg_src(ptr addrspace(6) %arg) {
main_body:
  %tmp9 = load ptr addrspace(8), ptr addrspace(6) %arg
  %tmp10 = call nsz float @llvm.amdgcn.struct.ptr.buffer.load.format.f32(ptr addrspace(8) %tmp9, i32 undef, i32 0, i32 0, i32 0) #1
  ret float %tmp10
}

; Function Attrs: nounwind readnone speculatable
declare float @llvm.amdgcn.interp.mov(i32, i32, i32, i32) #6

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #7

declare float @llvm.amdgcn.struct.ptr.buffer.load.format.f32(ptr addrspace(8), i32, i32, i32, i32) #7

!0 = !{}

attributes #0 = { nounwind }
attributes #1 = { nounwind "amdgpu-32bit-address-high-bits"="0" }
attributes #2 = { nounwind "amdgpu-32bit-address-high-bits"="1" }
attributes #3 = { nounwind "amdgpu-32bit-address-high-bits"="0xffff8000" }
attributes #4 = { nounwind "amdgpu-32bit-address-high-bits"="0xfffffff0" }
attributes #5 = { "InitialPSInputAddr"="45175" }
attributes #6 = { nounwind readnone speculatable }
attributes #7 = { nounwind memory(argmem: read) }
attributes #8 = { nounwind readnone }
