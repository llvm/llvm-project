; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

; CHECK: amdgpu_cs_chain void @amdgpu_cs_chain_cc
define amdgpu_cs_chain void @amdgpu_cs_chain_cc() {
entry:
  ret void
}

; CHECK: amdgpu_cs_chain_preserve void @amdgpu_cs_chain_preserve_cc
define amdgpu_cs_chain_preserve void @amdgpu_cs_chain_preserve_cc() {
entry:
  ret void
}
