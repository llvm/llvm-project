; RUN: llc -march=amdgcn -verify-machineinstrs< %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs< %s | FileCheck -check-prefix=SI %s

; Copy VGPR -> SGPR used twice as an instruction operand, which is then
; used in an REG_SEQUENCE that also needs to be handled.

; SI-LABEL: {{^}}test_dup_operands:
; SI: v_add_i32_e32
define void @test_dup_operands(<2 x i32> addrspace(1)* noalias %out, <2 x i32> addrspace(1)* noalias %in) {
  %a = load <2 x i32>, <2 x i32> addrspace(1)* %in
  %lo = extractelement <2 x i32> %a, i32 0
  %hi = extractelement <2 x i32> %a, i32 1
  %add = add i32 %lo, %lo
  %vec0 = insertelement <2 x i32> undef, i32 %add, i32 0
  %vec1 = insertelement <2 x i32> %vec0, i32 %hi, i32 1
  store <2 x i32> %vec1, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

