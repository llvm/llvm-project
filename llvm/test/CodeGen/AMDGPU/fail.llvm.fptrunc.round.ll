; RUN: not --crash llc -march=amdgcn -mcpu=gfx1030 -verify-machineinstrs -o /dev/null %s 2>&1 | FileCheck %s --ignore-case --check-prefix=FAIL
; RUN: not --crash llc -global-isel -march=amdgcn -mcpu=gfx1030 -verify-machineinstrs -o /dev/null %s 2>&1 | FileCheck %s --ignore-case --check-prefix=FAIL

define amdgpu_gs void @test_fptrunc_round_legalization(double %a, i32 %data0, <4 x i32> %data1, ptr addrspace(1) %out) {
; FAIL: LLVM ERROR: Cannot select
  %res = call half @llvm.fptrunc.round.f64(double %a, metadata !"round.upward")
  store half %res, ptr addrspace(1) %out, align 4
  ret void
}

declare half @llvm.fptrunc.round.f64(double, metadata)
