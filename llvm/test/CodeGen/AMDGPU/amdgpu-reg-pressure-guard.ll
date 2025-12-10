; RUN: opt -S -mtriple=amdgcn-amd-amdpal -passes=amdgpu-reg-pressure-estimator < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1200 -amdgpu-enable-reg-pressure-guard=true -amdgpu-reg-pressure-max-increase=1 -amdgpu-reg-pressure-min-baseline=1 < %s | FileCheck %s --check-prefix=GUARD

define amdgpu_cs void @test_reg_pressure(ptr addrspace(1) %out, ptr addrspace(1) %in) {
; CHECK: AMDGPU Register Pressure for function 'test_reg_pressure': {{[0-9]+}} VGPRs (IR-level estimate)
; GUARD-LABEL: test_reg_pressure:
; GUARD: global_store
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tidx = zext i32 %tid to i64
  %gep = getelementptr <4 x float>, ptr addrspace(1) %in, i64 %tidx
  %v = load <4 x float>, ptr addrspace(1) %gep, align 16
  %result = fadd <4 x float> %v, %v
  %outgep = getelementptr <4 x float>, ptr addrspace(1) %out, i64 %tidx
  store <4 x float> %result, ptr addrspace(1) %outgep, align 16
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
