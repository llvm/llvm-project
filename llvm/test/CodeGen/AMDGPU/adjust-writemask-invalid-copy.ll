; RUN: llc -mtriple=amdgcn -mcpu=tahiti < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}adjust_writemask_crash_0_nochain:
; GCN: image_get_lod v0, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0x2
; GCN-NOT: v1
; GCN-NOT: v0
; GCN: buffer_store_dword v0
define amdgpu_ps void @adjust_writemask_crash_0_nochain() #0 {
main_body:
  %tmp = call <2 x float> @llvm.amdgcn.image.getlod.1d.v2f32.f32(i32 3, float poison, <8 x i32> poison, <4 x i32> poison, i1 0, i32 0, i32 0)
  %tmp1 = bitcast <2 x float> %tmp to <2 x i32>
  %tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> poison, <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>
  %tmp3 = bitcast <4 x i32> %tmp2 to <4 x float>
  %tmp4 = extractelement <4 x float> %tmp3, i32 0
  store volatile float %tmp4, ptr addrspace(1) poison
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_crash_1_nochain:
; GCN: image_get_lod v0, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0x1
; GCN-NOT: v1
; GCN-NOT: v0
; GCN: buffer_store_dword v0
define amdgpu_ps void @adjust_writemask_crash_1_nochain() #0 {
main_body:
  %tmp = call <2 x float> @llvm.amdgcn.image.getlod.1d.v2f32.f32(i32 3, float poison, <8 x i32> poison, <4 x i32> poison, i1 0, i32 0, i32 0)
  %tmp1 = bitcast <2 x float> %tmp to <2 x i32>
  %tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> poison, <4 x i32> <i32 1, i32 0, i32 poison, i32 poison>
  %tmp3 = bitcast <4 x i32> %tmp2 to <4 x float>
  %tmp4 = extractelement <4 x float> %tmp3, i32 1
  store volatile float %tmp4, ptr addrspace(1) poison
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_crash_0_chain:
; GCN: image_sample v0, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0x2
; GCN-NOT: v1
; GCN-NOT: v0
; GCN: buffer_store_dword v0
define amdgpu_ps void @adjust_writemask_crash_0_chain() #0 {
main_body:
  %tmp = call <2 x float> @llvm.amdgcn.image.sample.1d.v2f32.f32(i32 3, float poison, <8 x i32> poison, <4 x i32> poison, i1 0, i32 0, i32 0)
  %tmp1 = bitcast <2 x float> %tmp to <2 x i32>
  %tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> poison, <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>
  %tmp3 = bitcast <4 x i32> %tmp2 to <4 x float>
  %tmp4 = extractelement <4 x float> %tmp3, i32 0
  store volatile float %tmp4, ptr addrspace(1) poison
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_crash_1_chain:
; GCN: image_sample v0, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0x1
; GCN-NOT: v1
; GCN-NOT: v0
; GCN: buffer_store_dword v0
define amdgpu_ps void @adjust_writemask_crash_1_chain() #0 {
main_body:
  %tmp = call <2 x float> @llvm.amdgcn.image.sample.1d.v2f32.f32(i32 3, float poison, <8 x i32> poison, <4 x i32> poison, i1 0, i32 0, i32 0)
  %tmp1 = bitcast <2 x float> %tmp to <2 x i32>
  %tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> poison, <4 x i32> <i32 1, i32 0, i32 poison, i32 poison>
  %tmp3 = bitcast <4 x i32> %tmp2 to <4 x float>
  %tmp4 = extractelement <4 x float> %tmp3, i32 1
  store volatile float %tmp4, ptr addrspace(1) poison
  ret void
}

define amdgpu_ps void @adjust_writemask_crash_0_v4() #0 {
main_body:
  %tmp = call <4 x float> @llvm.amdgcn.image.getlod.1d.v4f32.f32(i32 5, float poison, <8 x i32> poison, <4 x i32> poison, i1 0, i32 0, i32 0)
  %tmp1 = bitcast <4 x float> %tmp to <4 x i32>
  %tmp2 = shufflevector <4 x i32> %tmp1, <4 x i32> poison, <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>
  %tmp3 = bitcast <4 x i32> %tmp2 to <4 x float>
  %tmp4 = extractelement <4 x float> %tmp3, i32 0
  store volatile float %tmp4, ptr addrspace(1) poison
  ret void
}


declare <2 x float> @llvm.amdgcn.image.sample.1d.v2f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <2 x float> @llvm.amdgcn.image.getlod.1d.v2f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.getlod.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
