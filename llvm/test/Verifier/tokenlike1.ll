; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

define void @f() {
entry:
  %A = call target("amdgpu.stridemark") @llvm.ssa.copy(target("amdgpu.stridemark") poison)
  %B = call target("amdgpu.stridemark") @llvm.ssa.copy(target("amdgpu.stridemark") poison)
  br label %bb

bb:
  %phi = phi target("amdgpu.stridemark") [ %A, %bb ], [ %B, %entry]
; CHECK: PHI nodes cannot have token type!
  br label %bb
}
