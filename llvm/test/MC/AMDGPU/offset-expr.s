// RUN: llvm-mc -triple=amdgcn -mcpu=gfx906 -filetype=obj %s | llvm-objdump -d --mcpu=gfx906 - | FileCheck %s

// Check that the offset is correctly calculated.

BB0:
v_nop_e64
v_nop_e64
BB1:
v_nop_e64
BB2:
s_add_u32 vcc_lo, vcc_lo, (BB2-BB1)&4294967295
// CHECK: s_add_u32 vcc_lo, vcc_lo, 8   // 000000000018: 806AFF6A 00000008
s_addc_u32 vcc_hi, vcc_hi, (BB2-BB1)>>32
// CHECK: s_addc_u32 vcc_hi, vcc_hi, 0  // 000000000020: 826BFF6B 00000000
s_add_u32 vcc_lo, vcc_lo, (BB0-BB1)&4294967295
// CHECK: s_add_u32 vcc_lo, vcc_lo, -16 // 000000000028: 806AFF6A FFFFFFF0
s_addc_u32 vcc_hi, vcc_hi, (BB0-BB1)>>32
// CHECK: s_addc_u32 vcc_hi, vcc_hi, -1 // 000000000030: 826BFF6B FFFFFFFF
