; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes="amdgpu-promote-alloca-to-vector" -o - %s
; We don't really need to check anything here because with expensive check, this
; test case crashes. The correctness of the pass is beyond the scope.

define fastcc void @foo() {
entry:
  %det = alloca [4 x i32], align 16, addrspace(5)
  %trkltPosTmpYZ = alloca [2 x float], align 4, addrspace(5)
  %trkltCovTmp = alloca [3 x float], align 4, addrspace(5)
  ret void
}
