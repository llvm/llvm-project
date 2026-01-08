; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1250 < %s 2>&1 | FileCheck -check-prefix=ERR %s

; ERR: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.s.waitcnt

define amdgpu_kernel void @test_waitcnt_builtin() {
  call void @llvm.amdgcn.s.waitcnt(i32 0)
  ret void
}

declare void @llvm.amdgcn.s.waitcnt(i32)
