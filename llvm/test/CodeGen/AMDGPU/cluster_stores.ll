; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -debug-only=machine-scheduler < %s 2> %t | FileCheck --enable-var-scope --check-prefixes=CHECK,GCN %s
; RUN: FileCheck --enable-var-scope --check-prefixes=CHECK,DBG %s < %t
; REQUIRES: asserts

; CHECK-LABEL: {{^}}cluster_load_cluster_store:
define amdgpu_kernel void @cluster_load_cluster_store(i32* noalias %lb, i32* noalias %sb) {
bb:
; DBG: Cluster ld/st SU(1) - SU(2)

; DBG: Cluster ld/st SU([[L1:[0-9]+]]) - SU([[L2:[0-9]+]])
; DBG: Cluster ld/st SU([[L2]]) - SU([[L3:[0-9]+]])
; DBG: Cluster ld/st SU([[L3]]) - SU([[L4:[0-9]+]])
; GCN:      flat_load_dword [[LD1:v[0-9]+]], v[{{[0-9:]+}}]
; GCN-NEXT: flat_load_dword [[LD2:v[0-9]+]], v[{{[0-9:]+}}] offset:8
; GCN-NEXT: flat_load_dword [[LD3:v[0-9]+]], v[{{[0-9:]+}}] offset:16
; GCN-NEXT: flat_load_dword [[LD4:v[0-9]+]], v[{{[0-9:]+}}] offset:24
  %la0 = getelementptr inbounds i32, i32* %lb, i32 0
  %ld0 = load i32, i32* %la0
  %la1 = getelementptr inbounds i32, i32* %lb, i32 2
  %ld1 = load i32, i32* %la1
  %la2 = getelementptr inbounds i32, i32* %lb, i32 4
  %ld2 = load i32, i32* %la2
  %la3 = getelementptr inbounds i32, i32* %lb, i32 6
  %ld3 = load i32, i32* %la3

; DBG: Cluster ld/st SU([[S1:[0-9]+]]) - SU([[S2:[0-9]+]])
; DBG: Cluster ld/st SU([[S2]]) - SU([[S3:[0-9]+]])
; DBG: Cluster ld/st SU([[S3]]) - SU([[S4:[0-9]+]])
; GCN:      flat_store_dword v[{{[0-9:]+}}], [[LD1]]
; GCN-NEXT: flat_store_dword v[{{[0-9:]+}}], [[LD2]] offset:8
; GCN-NEXT: flat_store_dword v[{{[0-9:]+}}], [[LD3]] offset:16
; GCN-NEXT: flat_store_dword v[{{[0-9:]+}}], [[LD4]] offset:24
  %sa0 = getelementptr inbounds i32, i32* %sb, i32 0
  store i32 %ld0, i32* %sa0
  %sa1 = getelementptr inbounds i32, i32* %sb, i32 2
  store i32 %ld1, i32* %sa1
  %sa2 = getelementptr inbounds i32, i32* %sb, i32 4
  store i32 %ld2, i32* %sa2
  %sa3 = getelementptr inbounds i32, i32* %sb, i32 6
  store i32 %ld3, i32* %sa3

  ret void
}

; CHECK-LABEL: {{^}}cluster_load_valu_cluster_store:
define amdgpu_kernel void @cluster_load_valu_cluster_store(i32* noalias %lb, i32* noalias %sb) {
bb:
; DBG: Cluster ld/st SU(1) - SU(2)

; DBG: Cluster ld/st SU([[L1:[0-9]+]]) - SU([[L2:[0-9]+]])
; DBG: Cluster ld/st SU([[L2]]) - SU([[L3:[0-9]+]])
; DBG: Cluster ld/st SU([[L3]]) - SU([[L4:[0-9]+]])
; GCN:      flat_load_dword [[LD1:v[0-9]+]], v[{{[0-9:]+}}]
; GCN-NEXT: flat_load_dword [[LD2:v[0-9]+]], v[{{[0-9:]+}}] offset:8
; GCN-NEXT: flat_load_dword [[LD3:v[0-9]+]], v[{{[0-9:]+}}] offset:16
; GCN-NEXT: flat_load_dword [[LD4:v[0-9]+]], v[{{[0-9:]+}}] offset:24
  %la0 = getelementptr inbounds i32, i32* %lb, i32 0
  %ld0 = load i32, i32* %la0
  %la1 = getelementptr inbounds i32, i32* %lb, i32 2
  %ld1 = load i32, i32* %la1
  %la2 = getelementptr inbounds i32, i32* %lb, i32 4
  %ld2 = load i32, i32* %la2
  %la3 = getelementptr inbounds i32, i32* %lb, i32 6
  %ld3 = load i32, i32* %la3

; DBG: Cluster ld/st SU([[S1:[0-9]+]]) - SU([[S2:[0-9]+]])
; DBG: Cluster ld/st SU([[S2]]) - SU([[S3:[0-9]+]])
; DBG: Cluster ld/st SU([[S3]]) - SU([[S4:[0-9]+]])
; GCN:      v_add_u32_e32 [[ST2:v[0-9]+]], 1, [[LD2]]
; GCN:      flat_store_dword v[{{[0-9:]+}}], [[LD1]]
; GCN-NEXT: flat_store_dword v[{{[0-9:]+}}], [[ST2]] offset:8
; GCN-NEXT: flat_store_dword v[{{[0-9:]+}}], [[LD3]] offset:16
; GCN-NEXT: flat_store_dword v[{{[0-9:]+}}], [[LD4]] offset:24
  %sa0 = getelementptr inbounds i32, i32* %sb, i32 0
  store i32 %ld0, i32* %sa0
  %sa1 = getelementptr inbounds i32, i32* %sb, i32 2
  %add = add i32 %ld1, 1
  store i32 %add, i32* %sa1
  %sa2 = getelementptr inbounds i32, i32* %sb, i32 4
  store i32 %ld2, i32* %sa2
  %sa3 = getelementptr inbounds i32, i32* %sb, i32 6
  store i32 %ld3, i32* %sa3

  ret void
}

; Cluster loads from the same texture with different coordinates
; CHECK-LABEL: {{^}}cluster_image_load:
; DBG: {{^}}Cluster ld/st [[SU1:SU\([0-9]+\)]] - [[SU2:SU\([0-9]+\)]]
; DBG: {{^}}[[SU1]]: {{.*}} IMAGE_LOAD
; DBG: {{^}}[[SU2]]: {{.*}} IMAGE_LOAD
; GCN:      image_load v
; GCN-NEXT: image_load v
define amdgpu_ps void @cluster_image_load(<8 x i32> inreg %src, <8 x i32> inreg %dst, i32 %x, i32 %y) {
entry:
  %x1 = add i32 %x, 1
  %y1 = add i32 %y, 1
  %val1 = call <4 x float> @llvm.amdgcn.image.load.mip.2d.v4f32.i32(i32 15, i32 %x1, i32 %y1, i32 0, <8 x i32> %src, i32 0, i32 0)
  %x2 = add i32 %x, 2
  %y2 = add i32 %y, 2
  %val2 = call <4 x float> @llvm.amdgcn.image.load.mip.2d.v4f32.i32(i32 15, i32 %x2, i32 %y2, i32 0, <8 x i32> %src, i32 0, i32 0)
  %val = fadd fast <4 x float> %val1, %val2
  call void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float> %val, i32 15, i32 %x, i32 %y, <8 x i32> %dst, i32 0, i32 0)
  ret void
}

; Don't cluster loads from different textures
; CHECK-LABEL: {{^}}no_cluster_image_load:
; DBG-NOT: {{^}}Cluster ld/st
define amdgpu_ps void @no_cluster_image_load(<8 x i32> inreg %src1, <8 x i32> inreg %src2, <8 x i32> inreg %dst, i32 %x, i32 %y) {
entry:
  %val1 = call <4 x float> @llvm.amdgcn.image.load.mip.2d.v4f32.i32(i32 15, i32 %x, i32 %y, i32 0, <8 x i32> %src1, i32 0, i32 0)
  %val2 = call <4 x float> @llvm.amdgcn.image.load.mip.2d.v4f32.i32(i32 15, i32 %x, i32 %y, i32 0, <8 x i32> %src2, i32 0, i32 0)
  %val = fadd fast <4 x float> %val1, %val2
  call void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float> %val, i32 15, i32 %x, i32 %y, <8 x i32> %dst, i32 0, i32 0)
  ret void
}

; Cluster loads from the same texture and sampler with different coordinates
; CHECK-LABEL: {{^}}cluster_image_sample:
; DBG: {{^}}Cluster ld/st [[SU1:SU\([0-9]+\)]] - [[SU2:SU\([0-9]+\)]]
; DBG: {{^}}[[SU1]]: {{.*}} IMAGE_SAMPLE
; DBG: {{^}}[[SU2]]: {{.*}} IMAGE_SAMPLE
; GCN:      image_sample_d
; GCN-NEXT: image_sample_d
define amdgpu_ps void @cluster_image_sample(<8 x i32> inreg %src, <4 x i32> inreg %smp, <8 x i32> inreg %dst, i32 %x, i32 %y) {
entry:
  %s = sitofp i32 %x to float
  %t = sitofp i32 %y to float
  %s1 = fadd float %s, 1.0
  %t1 = fadd float %t, 1.0
  %val1 = call <4 x float> @llvm.amdgcn.image.sample.d.2d.v4f32.f32(i32 15, float %s1, float %t1, float 0.0, float 0.0, float 0.0, float 0.0, <8 x i32> %src, <4 x i32> %smp, i1 false, i32 0, i32 0)
  %s2 = fadd float %s, 2.0
  %t2 = fadd float %t, 2.0
  %val2 = call <4 x float> @llvm.amdgcn.image.sample.d.2d.v4f32.f32(i32 15, float %s2, float %t2, float 1.0, float 1.0, float 1.0, float 1.0, <8 x i32> %src, <4 x i32> %smp, i1 false, i32 0, i32 0)
  %val = fadd fast <4 x float> %val1, %val2
  call void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float> %val, i32 15, i32 %x, i32 %y, <8 x i32> %dst, i32 0, i32 0)
  ret void
}

declare <4 x float> @llvm.amdgcn.image.load.mip.2d.v4f32.i32(i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg)
declare <4 x float> @llvm.amdgcn.image.sample.d.2d.v4f32.f32(i32, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32)
declare void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float>, i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg)
