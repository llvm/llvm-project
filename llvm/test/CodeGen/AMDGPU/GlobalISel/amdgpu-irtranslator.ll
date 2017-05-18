; RUN: llc -march=amdgcn -mcpu=fiji -O0 -stop-after=irtranslator -global-isel %s -o - 2>&1 | FileCheck %s
; REQUIRES: global-isel
; This file checks that the translation from llvm IR to generic MachineInstr
; is correct.

; Tests for add.
; CHECK: name: addi32
; CHECK: {{%[0-9]+}}(s32) = G_ADD
define amdgpu_kernel void @addi32(i32 %arg1, i32 %arg2) {
  %res = add i32 %arg1, %arg2
  store i32 %res, i32 addrspace(1)* undef
  ret void
}
