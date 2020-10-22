// RUN: llvm-mc -arch=amdgcn -mcpu=tonga %s -filetype=obj | llvm-objdump -d --arch-name=amdgcn --mcpu=tonga - | FileCheck %s

	.text

	.amdgpu_hsa_kernel hello_world
hello_world:
  .amd_kernel_code_t
  .end_amd_kernel_code_t

	s_mov_b32 m0, 0x10000
	s_load_dwordx2 s[0:1], s[4:5], 0x8
	s_waitcnt lgkmcnt(0)
	s_add_u32 s0, s7, s0
BB0:
	v_add_u32_e32 v1, vcc, s0, v1
BB1:
	s_movk_i32 s0, 0x483
	v_cmp_ge_i32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[0:1], vcc
	v_lshlrev_b32_e32 v4, 2, v0
BB3:
	s_cbranch_execz 21
	s_mov_b64 s[2:3], exec
	s_mov_b64 s[10:11], exec
	v_mov_b32_e32 v3, v0
        s_endpgm

	.amdgpu_hsa_kernel hello_world2
hello_world2:
  .amd_kernel_code_t
  .end_amd_kernel_code_t

	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz 85
	s_load_dwordx4 s[8:11], s[4:5], 0x40
BB5:
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[10:11], 2, v[76:77]
	s_waitcnt lgkmcnt(0)
	v_add_u32_e32 v10, vcc, s8, v10
	v_mov_b32_e32 v6, s9
	v_addc_u32_e32 v11, vcc, v6, v11, vcc
	flat_load_dword v0, v[10:11]
	v_lshlrev_b32_e32 v6, 5, v8
	v_lshlrev_b32_e32 v7, 2, v7
        s_endpgm

// CHECK:  file format elf64-amdgpu
// CHECK:  Disassembly of section .text:
// CHECK:  <hello_world>:
// CHECK:  s_mov_b32 m0, 0x10000                                      // 000000000100: BEFC00FF 00010000
// CHECK:  s_load_dwordx2 s[0:1], s[4:5], 0x8                         // 000000000108: C0060002 00000008
// CHECK:  s_waitcnt lgkmcnt(0)                                       // 000000000110: BF8C007F
// CHECK:  s_add_u32 s0, s7, s0                                       // 000000000114: 80000007
// CHECK:  <BB0>:
// CHECK:  v_add_u32_e32 v1, vcc, s0, v1                              // 000000000118: 32020200
// CHECK:  <BB1>:
// CHECK:  s_movk_i32 s0, 0x483                                       // 00000000011C: B0000483
// CHECK:  v_cmp_ge_i32_e32 vcc, s0, v0                               // 000000000120: 7D8C0000
// CHECK:  s_and_saveexec_b64 s[0:1], vcc                             // 000000000124: BE80206A
// CHECK:  v_lshlrev_b32_e32 v4, 2, v0                                // 000000000128: 24080082
// CHECK:  <BB3>:
// CHECK:  s_cbranch_execz 21                                         // 00000000012C: BF880015
// CHECK:  s_mov_b64 s[2:3], exec                                     // 000000000130: BE82017E
// CHECK:  s_mov_b64 s[10:11], exec                                   // 000000000134: BE8A017E
// CHECK:  v_mov_b32_e32 v3, v0                                       // 000000000138: 7E060300
// CHECK:  s_endpgm                                                   // 00000000013C: BF810000

// CHECK:  <hello_world2>:
// CHECK:  s_and_saveexec_b64 s[0:1], vcc                             // 000000000240: BE80206A
// CHECK:  s_cbranch_execz 85                                         // 000000000244: BF880055
// CHECK:  s_load_dwordx4 s[8:11], s[4:5], 0x40                       // 000000000248: C00A0202 00000040
// CHECK:  <BB5>:
// CHECK:  v_ashrrev_i32_e32 v77, 31, v76                             // 000000000250: 229A989F
// CHECK:  v_lshlrev_b64 v[10:11], 2, v[76:77]                        // 000000000254: D28F000A 00029882
// CHECK:  s_waitcnt lgkmcnt(0)                                       // 00000000025C: BF8C007F
// CHECK:  v_add_u32_e32 v10, vcc, s8, v10                            // 000000000260: 32141408
// CHECK:  v_mov_b32_e32 v6, s9                                       // 000000000264: 7E0C0209
// CHECK:  v_addc_u32_e32 v11, vcc, v6, v11, vcc                      // 000000000268: 38161706
// CHECK:  flat_load_dword v0, v[10:11]                               // 00000000026C: DC500000 0000000A
// CHECK:  v_lshlrev_b32_e32 v6, 5, v8                                // 000000000274: 240C1085
// CHECK:  v_lshlrev_b32_e32 v7, 2, v7                                // 000000000278: 240E0E82
// CHECK:  s_endpgm                                                   // 00000000027C: BF810000
