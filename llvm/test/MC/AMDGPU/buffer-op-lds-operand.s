// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx950 --show-inst < %s | FileCheck %s

// CHECK: .amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx950"
buffer_load_dword off, s[8:11], 0 lds
// CHECK: buffer_load_dword off, s[8:11], 0 lds ; <MCInst #{{[0-9]+}} BUFFER_LOAD_DWORD_LDS_OFFSET
// CHECK-NEXT: ;  <MCOperand Reg:SGPR8_SGPR9_SGPR10_SGPR11>
// CHECK-NEXT: ;  <MCOperand Imm:0>
// CHECK-NEXT: ;  <MCOperand Imm:0>
// CHECK-NEXT: ;  <MCOperand Imm:0>
// CHECK-NEXT: ;  <MCOperand Imm:0>
// CHECK-NEXT: ;  <MCOperand Imm:0>>
buffer_load_dword v18, s[8:11], 0 offen lds
// CHECK: buffer_load_dword v18, s[8:11], 0 offen lds ; <MCInst #{{[0-9]+}} BUFFER_LOAD_DWORD_LDS_OFFEN
// CHECK-NEXT: ;  <MCOperand Reg:VGPR18>
// CHECK-NEXT: ;  <MCOperand Reg:SGPR8_SGPR9_SGPR10_SGPR11>
// CHECK-NEXT: ;  <MCOperand Imm:0>
// CHECK-NEXT: ;  <MCOperand Imm:0>
// CHECK-NEXT: ;  <MCOperand Imm:0>
// CHECK-NEXT: ;  <MCOperand Imm:0>
// CHECK-NEXT: ;  <MCOperand Imm:0>>
buffer_load_dword v18, s[8:11], 0 idxen lds
// CHECK: buffer_load_dword v18, s[8:11], 0 idxen lds ; <MCInst #{{[0-9]+}} BUFFER_LOAD_DWORD_LDS_IDXEN
// CHECK-NEXT: ;  <MCOperand Reg:VGPR18>
// CHECK-NEXT: ;  <MCOperand Reg:SGPR8_SGPR9_SGPR10_SGPR11>
// CHECK-NEXT: ;  <MCOperand Imm:0>
// CHECK-NEXT: ;  <MCOperand Imm:0>
// CHECK-NEXT: ;  <MCOperand Imm:0>
// CHECK-NEXT: ;  <MCOperand Imm:0>
// CHECK-NEXT: ;  <MCOperand Imm:0>>
