; RUN: llc -march=amdgcn -mcpu=gfx900 < %s | FileCheck --check-prefix=GCN %s

; This is used to crash due to mismatch of MMO target flags when folding
; a LOAD SDNodes with different flags.

; GCN-LABEL: {{^}}test_load_folding_mmo_flags:
; GCN: global_load_dwordx2
define amdgpu_kernel void @test_load_folding_mmo_flags(<2 x float> addrspace(1)* %arg) {
entry:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %arrayidx = getelementptr inbounds <2 x float>, <2 x float> addrspace(1)* %arg, i32 %id
  %i1 = bitcast <2 x float> addrspace(1)* %arrayidx to i64 addrspace(1)*
  %i2 = getelementptr <2 x float>, <2 x float> addrspace(1)* %arrayidx, i64 0, i32 0
  %i3 = load float, float addrspace(1)* %i2, align 4
  %idx = getelementptr inbounds <2 x float>, <2 x float> addrspace(1)* %arrayidx, i64 0, i32 1
  %i4 = load float, float addrspace(1)* %idx, align 4
  %i5 = load i64, i64 addrspace(1)* %i1, align 4, !amdgpu.noclobber !0
  store i64 %i5, i64 addrspace(1)* undef, align 4
  %mul = fmul float %i3, %i4
  store float %mul, float addrspace(1)* undef, align 4
  unreachable
}

declare i32 @llvm.amdgcn.workitem.id.x()

!0 = !{}
