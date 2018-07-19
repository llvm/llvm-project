; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}adjust_writemask_crash_0_nochain:
; GCN: image_get_lod v0, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0x2
; GCN-NOT: v1
; GCN-NOT: v0
; GCN: buffer_store_dword v0
define amdgpu_ps void @adjust_writemask_crash_0_nochain() #0 {
main_body:
  %tmp = call <2 x float> @llvm.amdgcn.image.getlod.v2f32.v2f32.v8i32(<2 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 3, i1 false, i1 false, i1 false, i1 false, i1 false)
  %tmp1 = bitcast <2 x float> %tmp to <2 x i32>
  %tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %tmp3 = bitcast <4 x i32> %tmp2 to <4 x float>
  %tmp4 = extractelement <4 x float> %tmp3, i32 0
  store volatile float %tmp4, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_crash_1_nochain:
; GCN: image_get_lod v0, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0x1
; GCN-NOT: v1
; GCN-NOT: v0
; GCN: buffer_store_dword v0
define amdgpu_ps void @adjust_writemask_crash_1_nochain() #0 {
main_body:
  %tmp = call <2 x float> @llvm.amdgcn.image.getlod.v2f32.v2f32.v8i32(<2 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 3, i1 false, i1 false, i1 false, i1 false, i1 false)
  %tmp1 = bitcast <2 x float> %tmp to <2 x i32>
  %tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> undef, <4 x i32> <i32 1, i32 0, i32 undef, i32 undef>
  %tmp3 = bitcast <4 x i32> %tmp2 to <4 x float>
  %tmp4 = extractelement <4 x float> %tmp3, i32 1
  store volatile float %tmp4, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_crash_0_chain:
; GCN: image_sample v0, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0x2
; GCN-NOT: v1
; GCN-NOT: v0
; GCN: buffer_store_dword v0
define amdgpu_ps void @adjust_writemask_crash_0_chain() #0 {
main_body:
  %tmp = call <2 x float> @llvm.amdgcn.image.sample.v2f32.v2f32.v8i32(<2 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 3, i1 false, i1 false, i1 false, i1 false, i1 false)
  %tmp1 = bitcast <2 x float> %tmp to <2 x i32>
  %tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %tmp3 = bitcast <4 x i32> %tmp2 to <4 x float>
  %tmp4 = extractelement <4 x float> %tmp3, i32 0
  store volatile float %tmp4, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}adjust_writemask_crash_1_chain:
; GCN: image_sample v0, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} dmask:0x1
; GCN-NOT: v1
; GCN-NOT: v0
; GCN: buffer_store_dword v0
define amdgpu_ps void @adjust_writemask_crash_1_chain() #0 {
main_body:
  %tmp = call <2 x float> @llvm.amdgcn.image.sample.v2f32.v2f32.v8i32(<2 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 3, i1 false, i1 false, i1 false, i1 false, i1 false)
  %tmp1 = bitcast <2 x float> %tmp to <2 x i32>
  %tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> undef, <4 x i32> <i32 1, i32 0, i32 undef, i32 undef>
  %tmp3 = bitcast <4 x i32> %tmp2 to <4 x float>
  %tmp4 = extractelement <4 x float> %tmp3, i32 1
  store volatile float %tmp4, float addrspace(1)* undef
  ret void
}

define amdgpu_ps void @adjust_writemask_crash_0_v4() #0 {
main_body:
  %tmp = call <4 x float> @llvm.amdgcn.image.getlod.v4f32.v2f32.v8i32(<2 x float> undef, <8 x i32> undef, <4 x i32> undef, i32 5, i1 false, i1 false, i1 false, i1 false, i1 false)
  %tmp1 = bitcast <4 x float> %tmp to <4 x i32>
  %tmp2 = shufflevector <4 x i32> %tmp1, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %tmp3 = bitcast <4 x i32> %tmp2 to <4 x float>
  %tmp4 = extractelement <4 x float> %tmp3, i32 0
  store volatile float %tmp4, float addrspace(1)* undef
  ret void
}


declare <2 x float> @llvm.amdgcn.image.sample.v2f32.v2f32.v8i32(<2 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1
declare <2 x float> @llvm.amdgcn.image.getlod.v2f32.v2f32.v8i32(<2 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1
declare <4 x float> @llvm.amdgcn.image.getlod.v4f32.v2f32.v8i32(<2 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
