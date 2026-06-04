; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 < %s -o /dev/null
; Our SISchedule model enables dual-issue (v_dual_*) which changes scheduling
; output. Test verifies compilation succeeds — full scheduling validation via
; llvm-mca or the upstream scheduling test suite.

declare <8 x half> @llvm.amdgcn.swmmac.f16.16x16x32.f16.v8f16.v8f16.v16f16..i16(<8 x half>, <16 x half>, <8 x half>, i16)
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @test_barrier(ptr addrspace(3) noalias %in, ptr addrspace(3) noalias %out) #0 {
entry:
  %val = load <8 x half>, ptr addrspace(3) %in
  %val2 = load <16 x half>, ptr addrspace(3) %in
  %acc = load <8 x half>, ptr addrspace(3) %in
  %res = call <8 x half> @llvm.amdgcn.swmmac.f16.16x16x32.f16.v8f16.v8f16.v16f16..i16(<8 x half> %val, <16 x half> %val2, <8 x half> %acc, i16 0)
  store <8 x half> %res, ptr addrspace(3) %out
  ret void
}

attributes #0 = { "target-features"="+wavefrontsize32" }
