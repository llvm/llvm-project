; RUN: llc -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck --enable-var-scope --check-prefixes=GCN,GFX908 %s
; RUN: llc -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck --enable-var-scope --check-prefixes=GCN,GFX90A %s
; RUN: llc -global-isel -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck --enable-var-scope --check-prefixes=GCN,GFX90A %s

declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float, float, <32 x float>, i32, i32, i32)

; GCN-LABEL: {{^}}test_mfma_f32_32x32x1f32_vgpr:
; GFX908: v_mfma_f32_32x32x1{{.*}} a[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9:]+}}, a[{{[0-9:]+}}]
; GFX90A: v_mfma_f32_32x32x1{{.*}} v[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9:]+}}, v[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x1f32_vgpr(<32 x float> addrspace(1)* %arg) #0 {
bb:
  %in.1 = load <32 x float>, <32 x float> addrspace(1)* %arg
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 0, i32 0, i32 0)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x1f32_agpr:
; GCN: v_mfma_f32_32x32x1{{.*}} a[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9:]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x1f32_agpr(<32 x float> addrspace(1)* %arg) #1 {
bb:
  %in.1 = load <32 x float>, <32 x float> addrspace(1)* %arg
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 0, i32 0, i32 0)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x1f32_inline_asm_virtual_agpr:
; GCN: v_mfma_f32_32x32x1{{.*}} a[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9:]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x1f32_inline_asm_virtual_agpr(<32 x float> addrspace(1)* %arg) #0 {
bb:
  %acc = call i32 asm sideeffect "; def $0", "={a0}"()
  %in.1 = load <32 x float>, <32 x float> addrspace(1)* %arg
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 0, i32 0, i32 0)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x1f32_inline_asm_phys_agpr:
; GCN: v_mfma_f32_32x32x1{{.*}} a[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9:]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x1f32_inline_asm_phys_agpr(<32 x float> addrspace(1)* %arg) #0 {
bb:
  call void asm sideeffect "; use $0", "{a[100:131]}"(<32 x float> undef)
  %in.1 = load <32 x float>, <32 x float> addrspace(1)* %arg
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 0, i32 0, i32 0)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x1f32_inline_asm_no_agprs:
; GFX908: v_mfma_f32_32x32x1{{.*}} a[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9:]+}}, a[{{[0-9:]+}}]
; GFX90A: v_mfma_f32_32x32x1{{.*}} v[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9:]+}}, v[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x1f32_inline_asm_no_agprs(<32 x float> addrspace(1)* %arg) #0 {
bb:
  %acc = call i32 asm sideeffect "; def $0", "={v0}"()
  %in.1 = load <32 x float>, <32 x float> addrspace(1)* %arg
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 0, i32 0, i32 0)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x1f32_call:
; GCN: v_mfma_f32_32x32x1{{.*}} a[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9:]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x1f32_call(<32 x float> addrspace(1)* %arg) #0 {
bb:
  call void @foo()
  %in.1 = load <32 x float>, <32 x float> addrspace(1)* %arg
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 0, i32 0, i32 0)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %arg
  ret void
}

; We could avoid scan to find calls since we see these during lowering before selection.
; However, in SDag lowering and selection is done block by block, so it would only work
; in Global ISel.

; GCN-LABEL: {{^}}test_mfma_f32_32x32x1f32_call_multi_bb:
; GCN: v_mfma_f32_32x32x1{{.*}} a[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9:]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x1f32_call_multi_bb(<32 x float> addrspace(1)* %arg) #0 {
bb1:
  %in.1 = load <32 x float>, <32 x float> addrspace(1)* %arg
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 1, i32 2, i32 3)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %arg
  br i1 undef, label %bb2, label %bb3
  br label %bb2

bb2:
  call void @foo()
  br label %bb3

bb3:
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x1f32_nonentry:
; GCN: v_mfma_f32_32x32x1{{.*}} a[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9:]+}}, a[{{[0-9:]+}}]
define void @test_mfma_f32_32x32x1f32_nonentry(<32 x float> addrspace(1)* %arg) #0 {
bb:
  %in.1 = load <32 x float>, <32 x float> addrspace(1)* %arg
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 0, i32 0, i32 0)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %arg
  ret void
}

declare void @foo()

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" "amdgpu-waves-per-eu"="2" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,256" }
