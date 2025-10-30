// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1100 --show-inst < %s | FileCheck %s

// CHECK: .amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
buffer_load_dwordx4 v[0:3], v0, s[0:3], 0, offen offset:4092 slc
// CHECK: buffer_load_b128 v[0:3], v0, s[0:3], 0 offen offset:4092 slc ; <MCInst #{{[0-9]+}} BUFFER_LOAD_DWORDX4_OFFEN_gfx11
// CHECK-NEXT: ;  <MCOperand Reg:10104>
// CHECK-NEXT: ;  <MCOperand Reg:486>
// CHECK-NEXT: ;  <MCOperand Reg:7754>
// CHECK-NEXT: ;  <MCOperand Imm:0>
// CHECK-NEXT: ;  <MCOperand Imm:4092>
// CHECK-NEXT: ;  <MCOperand Imm:2>
// CHECK-NEXT: ;  <MCOperand Imm:0>>
buffer_store_dword v0, v1, s[0:3], 0 offen slc
// CHECK: buffer_store_b32 v0, v1, s[0:3], 0 offen slc ; <MCInst #{{[0-9]+}} BUFFER_STORE_DWORD_OFFEN_gfx11
// CHECK-NEXT: ;  <MCOperand Reg:486>
// CHECK-NEXT: ;  <MCOperand Reg:487>
// CHECK-NEXT: ;  <MCOperand Reg:7754>
// CHECK-NEXT: ;  <MCOperand Imm:0>
// CHECK-NEXT: ;  <MCOperand Imm:0>
// CHECK-NEXT: ;  <MCOperand Imm:2>
// CHECK-NEXT: ;  <MCOperand Imm:0>>

; tbuffer ops use autogenerate asm parsers
tbuffer_load_format_xyzw v[0:3], v0, s[0:3], 0 format:[BUF_FMT_32_32_SINT] offen offset:4092 slc
// CHECK: tbuffer_load_format_xyzw v[0:3], v0, s[0:3], 0 format:[BUF_FMT_32_32_SINT] offen offset:4092 slc ; <MCInst #{{[0-9]+}} TBUFFER_LOAD_FORMAT_XYZW_OFFEN_gfx11
// CHECK-NEXT: ;  <MCOperand Reg:10104>
// CHECK-NEXT: ;  <MCOperand Reg:486>
// CHECK-NEXT: ;  <MCOperand Reg:7754>
// CHECK-NEXT: ;  <MCOperand Imm:0>
// CHECK-NEXT: ;  <MCOperand Imm:4092>
// CHECK-NEXT: ;  <MCOperand Imm:49>
// CHECK-NEXT: ;  <MCOperand Imm:2>
// CHECK-NEXT: ;  <MCOperand Imm:0>>
tbuffer_store_d16_format_x v0, v1, s[0:3], 0 format:[BUF_FMT_10_10_10_2_SNORM] offen slc
// CHECK: tbuffer_store_d16_format_x v0, v1, s[0:3], 0 format:[BUF_FMT_10_10_10_2_SNORM] offen slc ; <MCInst #{{[0-9]+}} TBUFFER_STORE_FORMAT_D16_X_OFFEN_gfx11
// CHECK-NEXT: ;  <MCOperand Reg:486>
// CHECK-NEXT: ;  <MCOperand Reg:487>
// CHECK-NEXT: ;  <MCOperand Reg:7754>
// CHECK-NEXT: ;  <MCOperand Imm:0>
// CHECK-NEXT: ;  <MCOperand Imm:0>
// CHECK-NEXT: ;  <MCOperand Imm:33>
// CHECK-NEXT: ;  <MCOperand Imm:2>
// CHECK-NEXT: ;  <MCOperand Imm:0>>
