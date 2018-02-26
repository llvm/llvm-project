; RUN: llc -o - %s -march=amdgcn -mcpu=verde -verify-machineinstrs -stop-after expand-isel-pseudos | FileCheck %s
; This test verifies that the instruction selection will add the implicit
; register operands in the correct order when modifying the opcode of an
; instruction to V_ADD_I32_e32.

; CHECK: %{{[0-9]+}}:vgpr_32 = V_ADD_I32_e32 %{{[0-9]+}}, %{{[0-9]+}}, implicit-def %vcc, implicit %exec

define amdgpu_kernel void @test(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
entry:
  %b_ptr = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %a = load volatile i32, i32 addrspace(1)* %in
  %b = load volatile i32, i32 addrspace(1)* %b_ptr
  %result = add i32 %a, %b
  store i32 %result, i32 addrspace(1)* %out
  ret void
}
