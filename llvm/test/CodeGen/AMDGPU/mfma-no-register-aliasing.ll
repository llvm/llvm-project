; RUN: llc -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GREEDY,GREEDY908 %s
; RUN: llc -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GREEDY,GREEDY90A %s
; RUN: llc -march=amdgcn -mcpu=gfx940 -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GREEDY,GREEDY90A %s
; RUN: llc -global-isel -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GREEDY,GREEDY90A-GISEL %s
; RUN: llc -march=amdgcn -mcpu=gfx90a -sgpr-regalloc=fast -vgpr-regalloc=fast -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,FAST %s

; Check that Dst and SrcC of MFMA instructions reading more than 4 registers as SrcC
; is either completely disjoint or exactly the same, but does not alias.

declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float, float, <32 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float, float, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float, float, <4 x float>, i32, i32, i32)

; GCN-LABEL: {{^}}test_mfma_f32_32x32x1f32:
; GREEDY: v_mfma_f32_32x32x1{{.*}} a[0:31], v{{[0-9]+}}, v{{[0-9]+}}, a[0:31]
; GREEDY: v_mfma_f32_32x32x1{{.*}} a[32:63], v{{[0-9]+}}, v{{[0-9]+}}, a[0:31]
; FAST:   v_mfma_f32_32x32x1{{.*}} a[64:95], v{{[0-9]+}}, v{{[0-9]+}}, a[64:95]
; FAST:   v_mfma_f32_32x32x1{{.*}} a[32:63], v{{[0-9]+}}, v{{[0-9]+}}, a[64:95]
; GCN:    v_mfma_f32_32x32x1{{.*}} a[0:31], v{{[0-9]+}}, v{{[0-9]+}}, a[0:31]
define amdgpu_kernel void @test_mfma_f32_32x32x1f32(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <32 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 0, i32 0, i32 0)
  %mai.2 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %mai.1, i32 0, i32 0, i32 0)
  %tmp.1 = shufflevector <32 x float> %mai.2, <32 x float> %mai.1, <32 x i32> <i32 32, i32 33, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29>
  %mai.3 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %tmp.1, i32 0, i32 0, i32 0)
  store <32 x float> %mai.3, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x1f32:
; GREEDY908: v_mfma_f32_16x16x1{{.*}} a[16:31], v{{[0-9]+}}, v{{[0-9]+}}, a[16:31]
; GREEDY908: v_mfma_f32_16x16x1{{.*}} a[0:15], v{{[0-9]+}}, v{{[0-9]+}}, a[16:31]

; GREEDY90A: v_mfma_f32_16x16x1{{.*}} a[16:31], v{{[0-9]+}}, v{{[0-9]+}}, a[16:31]
; GREEDY90A: v_mfma_f32_16x16x1{{.*}} a[0:15], v{{[0-9]+}}, v{{[0-9]+}}, a[16:31]

; GREEDY90A-GISEL: v_mfma_f32_16x16x1{{.*}} a[0:15], v{{[0-9]+}}, v{{[0-9]+}}, a[0:15]
; GREEDY90A-GISEL: v_mfma_f32_16x16x1{{.*}} a[16:31], v{{[0-9]+}}, v{{[0-9]+}}, a[0:15]

; FAST:   v_mfma_f32_16x16x1{{.*}} a[32:47], v{{[0-9]+}}, v{{[0-9]+}}, a[32:47]
; FAST:   v_mfma_f32_16x16x1{{.*}} a[16:31], v{{[0-9]+}}, v{{[0-9]+}}, a[32:47]
; GCN:    v_mfma_f32_16x16x1{{.*}} a[0:15], v{{[0-9]+}}, v{{[0-9]+}}, a[0:15]
define amdgpu_kernel void @test_mfma_f32_16x16x1f32(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float 1.0, float 2.0, <16 x float> %in.1, i32 0, i32 0, i32 0)
  %mai.2 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float 1.0, float 2.0, <16 x float> %mai.1, i32 0, i32 0, i32 0)
  %tmp.1 = shufflevector <16 x float> %mai.2, <16 x float> %mai.1, <16 x i32> <i32 16, i32 17, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13>
  %mai.3 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float 1.0, float 2.0, <16 x float> %tmp.1, i32 0, i32 0, i32 0)
  store <16 x float> %mai.3, ptr addrspace(1) %arg
  ret void
}

; This instruction allows the overlap since it only read 4 registers.

; GCN-LABEL: {{^}}test_mfma_f32_4x4x1f32:
; GREEDY: v_mfma_f32_4x4x1{{.*}} a[0:3], v{{[0-9]+}}, v{{[0-9]+}}, a[0:3]
; GREEDY: v_mfma_f32_4x4x1{{.*}} a[2:5], v{{[0-9]+}}, v{{[0-9]+}}, a[0:3]
; FAST:   v_mfma_f32_4x4x1{{.*}} a[8:11], v{{[0-9]+}}, v{{[0-9]+}}, a[0:3]
; FAST:   v_mfma_f32_4x4x1{{.*}} a[4:7], v{{[0-9]+}}, v{{[0-9]+}}, a[8:11]
; GCN:    v_mfma_f32_4x4x1{{.*}} a[0:3], v{{[0-9]+}}, v{{[0-9]+}}, a[0:3]
define amdgpu_kernel void @test_mfma_f32_4x4x1f32(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> %in.1, i32 0, i32 0, i32 0)
  %mai.2 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> %mai.1, i32 0, i32 0, i32 0)
  %tmp.1 = shufflevector <4 x float> %mai.1, <4 x float> %mai.2, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %mai.3 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> %tmp.1, i32 0, i32 0, i32 0)
  store <4 x float> %mai.3, ptr addrspace(1) %arg
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" }
