; RUN: not --crash llc -mtriple=amdgpu13.10 < %s 2>&1 | FileCheck -check-prefix=ERR %s

; ERR: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.s.wait.bvhcnt

define amdgpu_ps void @test_bvhcnt() {
  call void @llvm.amdgcn.s.wait.bvhcnt(i16 0)
  ret void
}

declare void @llvm.amdgcn.s.wait.bvhcnt(i16)
