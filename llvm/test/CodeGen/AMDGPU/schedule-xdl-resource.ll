; RUN: llc -march=amdgcn -mcpu=gfx908 -debug-only=machine-scheduler -verify-machineinstrs < %s 2>&1 | FileCheck -enable-var-scope %s
; REQUIRES: asserts

declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x4f16(<4 x half>, <4 x half>, <32 x float>, i32, i32, i32)

; CHECK: CritRes: {{[0-9]+}} HWXDL
; CHECK: Picking: Cand SU([[nid:[0-9]+]]) RES-DEMAND
; CHECK: Scheduling SU([[nid]]) {{.*}} V_MFMA_F32_32X32X4F16
define amdgpu_kernel void @schedule-xdl-resource(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(3) %lds, i32 %stride) #0 {
  %in_ptr.1 = getelementptr <32 x float>, ptr addrspace(1) %in, i32 %stride
  %in_ptr.2 = getelementptr <32 x float>, ptr addrspace(1) %in_ptr.1, i32 %stride
  %in_ptr.3 = getelementptr <32 x float>, ptr addrspace(1) %in_ptr.2, i32 %stride
  %in.load.1 = load <32 x float>, ptr addrspace (1) %in_ptr.1
  %in.load.2 = load <32 x float>, ptr addrspace (1) %in_ptr.2
  %in.load.3 = load <32 x float>, ptr addrspace (1) %in_ptr.3
  %lds_ptr.1 = getelementptr <4 x half>, ptr addrspace(3) %lds, i32 %stride
  %lds_ptr.2 = getelementptr <4 x half>, ptr addrspace(3) %lds_ptr.1, i32 %stride
  %lds_ptr.3 = getelementptr <4 x half>, ptr addrspace(3) %lds_ptr.2, i32 %stride
  %lds.load.1 = load <4 x half>, ptr addrspace(3) %lds_ptr.1
  %lds.load.2 = load <4 x half>, ptr addrspace(3) %lds_ptr.2
  %lds.load.3 = load <4 x half>, ptr addrspace(3) %lds_ptr.3
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x4f16(<4 x half> %lds.load.1, <4 x half> %lds.load.1, <32 x float> %in.load.1, i32 1, i32 1, i32 1)
  %mai.2 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x4f16(<4 x half> %lds.load.2, <4 x half> %lds.load.2, <32 x float> %in.load.2, i32 1, i32 1, i32 1)
  %mai.3 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x4f16(<4 x half> %lds.load.3, <4 x half> %lds.load.3, <32 x float> %in.load.3, i32 1, i32 1, i32 1)
  %mai.4 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x4f16(<4 x half> %lds.load.1, <4 x half> %lds.load.1, <32 x float> %in.load.1, i32 2, i32 2, i32 2)
  %mai.5 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x4f16(<4 x half> %lds.load.2, <4 x half> %lds.load.2, <32 x float> %in.load.2, i32 2, i32 2, i32 2)
  %mai.6 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x4f16(<4 x half> %lds.load.3, <4 x half> %lds.load.3, <32 x float> %in.load.3, i32 2, i32 2, i32 2)
  %out_ptr.1 = getelementptr <32 x float>, ptr addrspace(1) %out, i32 %stride
  %out_ptr.2 = getelementptr <32 x float>, ptr addrspace(1) %out_ptr.1, i32 %stride
  %out_ptr.3 = getelementptr <32 x float>, ptr addrspace(1) %out_ptr.2, i32 %stride
  %out_ptr.4 = getelementptr <32 x float>, ptr addrspace(1) %out_ptr.3, i32 %stride
  %out_ptr.5 = getelementptr <32 x float>, ptr addrspace(1) %out_ptr.4, i32 %stride
  %out_ptr.6 = getelementptr <32 x float>, ptr addrspace(1) %out_ptr.5, i32 %stride
  store <32 x float> %mai.1, ptr addrspace(1) %out_ptr.1
  store <32 x float> %mai.2, ptr addrspace(1) %out_ptr.2
  store <32 x float> %mai.3, ptr addrspace(1) %out_ptr.3
  store <32 x float> %mai.4, ptr addrspace(1) %out_ptr.4
  store <32 x float> %mai.5, ptr addrspace(1) %out_ptr.5
  store <32 x float> %mai.6, ptr addrspace(1) %out_ptr.6

  ret void
}

attributes #0 = { nounwind "amdgpu-waves-per-eu"="1,1" }
