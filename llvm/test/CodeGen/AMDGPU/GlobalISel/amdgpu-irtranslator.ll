; RUN: llc -mtriple=amdgcn -mcpu=fiji -O0 -stop-after=irtranslator -global-isel %s -o - 2>&1 | FileCheck %s

; This file checks that the translation from llvm IR to generic
; MachineInstr is correct.

; Tests for add.
; CHECK: name: addi32
; CHECK: {{%[0-9]+}}:_(i32) = G_ADD
define amdgpu_kernel void @addi32(i32 %arg1, i32 %arg2) {
  %res = add i32 %arg1, %arg2
  store i32 %res, ptr addrspace(1) undef
  ret void
}
