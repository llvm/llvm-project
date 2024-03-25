// RUN: not llvm-mc -triple=amdgcn -mcpu=bonaire %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=tonga   %s 2>&1 | FileCheck --implicit-check-not=error: %s

// GENERIC LIMITATIONS ON VOP FORMATS: CONSTANT BUS RESTRICTIONS

//=====================================================
// v_movreld_b32: implicitly reads m0 (VOP1/VOP3)

v_movreld_b32 v0, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_movreld_b32 v0, flat_scratch_lo
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_movreld_b32 v0, flat_scratch_hi
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_movreld_b32 v0, vcc_lo
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_movreld_b32 v0, vcc_hi
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_movreld_b32 v0, exec_lo
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_movreld_b32 v0, exec_hi
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_movreld_b32 v0, ttmp0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_movreld_b32 v0, ttmp1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_movreld_b32 v0, 123
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_movreld_b32_e64 v0, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_movreld_b32_e64 v0, flat_scratch_lo
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_movreld_b32_e64 v0, flat_scratch_hi
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

//=====================================================
// v_div_fmas: implicitly read VCC (VOP3)

v_div_fmas_f32 v0, s1, s1, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_div_fmas_f32 v0, v2, v3, -s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_div_fmas_f32 v0, v1, s2, |v3|
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_div_fmas_f32 v0, v1, -v2, -s3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_div_fmas_f32 v0, v1, flat_scratch_lo, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_div_fmas_f32 v0, v1, v2, flat_scratch_hi
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_div_fmas_f32 v0, v1, v2, m0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_div_fmas_f32 v0, v1, ttmp2, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_div_fmas_f64 v[0:1], s[2:3], v[4:5], v[6:7]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_div_fmas_f64 v[0:1], v[2:3], s[4:5], v[6:7]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_div_fmas_f64 v[0:1], v[2:3], v[4:5], s[6:7]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_div_fmas_f64 v[0:1], v[2:3], v[4:5], ttmp[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_div_fmas_f64 v[0:1], v[2:3], v[4:5], flat_scratch
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_div_fmas_f64 v[0:1], v[2:3], v[4:5], exec
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

//=====================================================
// v_cndmask_b32: implicitly reads VCC (VOP2)

v_cndmask_b32 v0, s1, v2, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cndmask_b32 v0, flat_scratch_lo, v2, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cndmask_b32 v0, flat_scratch_hi, v2, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cndmask_b32 v0, exec_lo, v2, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cndmask_b32 v0, exec_hi, v2, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

//=====================================================
// v_cndmask_b32_e64: VOP3, no implicit reads

v_cndmask_b32_e64 v0, s1, v2, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cndmask_b32_e64 v0, flat_scratch_lo, v2, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cndmask_b32_e64 v0, flat_scratch_hi, v2, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cndmask_b32_e64 v0, s1, v2, flat_scratch
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cndmask_b32_e64 v0, s0, v2, s[0:1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cndmask_b32_e64 v0, v2, s0, s[0:1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cndmask_b32_e64 v0, s0, s0, s[0:1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cndmask_b32_e64 v0, s1, v2, s[0:1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cndmask_b32_e64 v0, v2, s1, s[0:1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cndmask_b32_e64 v0, s1, s1, s[0:1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cndmask_b32_e64 v0, s1, v2, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cndmask_b32_e64 v0, v2, s1, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cndmask_b32_e64 v0, s1, s1, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

//=====================================================
// v_addc_u32: implicitly reads VCC (VOP2 only!)

v_addc_u32 v0, vcc, s0, v0, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_addc_u32 v0, vcc, flat_scratch_lo, v0, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_addc_u32 v0, vcc, flat_scratch_hi, v0, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_addc_u32 v0, vcc, exec_lo, v0, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_addc_u32 v0, vcc, exec_hi, v0, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

//=====================================================
// v_addc_u32_e64: no implicit read in VOP3

v_addc_u32_e64 v0, s[0:1], s2, v2, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_addc_u32_e64 v0, s[0:1], v2, s2, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_addc_u32_e64 v0, s[0:1], s2, s2, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_addc_u32_e64 v0, s[0:1], s0, v2, s[0:1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_addc_u32_e64 v0, s[0:1], v2, s0, s[0:1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_addc_u32_e64 v0, s[0:1], s0, s0, s[0:1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_addc_u32_e64 v0, s[0:1], s2, v2, s[0:1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_addc_u32_e64 v0, s[0:1], v2, s2, s[0:1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_addc_u32_e64 v0, s[0:1], s2, s2, s[0:1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

//=====================================================
// VOP1 w/o implicit reads have no negative test cases on constant bus use
// VOPC has no negative test cases on constant bus use

//=====================================================
// madak/madmk: a special case for VOP2 w/o implicit reads

v_madak_f32 v0, s0, v0, 0x11213141
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_madak_f32 v0, flat_scratch_lo, v0, 0x11213141
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_madak_f32 v0, flat_scratch_hi, v0, 0x11213141
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_madak_f32 v0, exec_lo, v0, 0x11213141
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_madak_f32 v0, exec_hi, v0, 0x11213141
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_madak_f32 v0, vcc_lo, v0, 0x11213141
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_madak_f32 v0, vcc_hi, v0, 0x11213141
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

//=====================================================
// VOP3 w/o implicit reads

v_mad_f32 v0, s0, s1, s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_mad_f32 v0, s1, s0, s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_mad_f32 v0, s0, s0, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_mad_f32 v0, s0, s0, flat_scratch_lo
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

//=====================================================
// VOP2_e64:

v_add_f32_e64 v0, s0, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_add_f32_e64 v0, s0, flat_scratch_lo
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_add_f32_e64 v0, flat_scratch_hi, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_add_f32_e64 v0, flat_scratch_hi, m0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_add_f64 v[0:1], s[0:1], s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_add_f64 v[0:1], s[0:1], flat_scratch
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_add_f64 v[0:1], vcc, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

//=====================================================
// VOPC_e64:

v_cmp_eq_f32_e64 s[0:1], s0, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cmp_eq_f32_e64 s[0:1], s0, flat_scratch_lo
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cmp_eq_f32_e64 s[0:1], flat_scratch_hi, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cmp_eq_f32_e64 s[0:1], s0, m0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cmp_eq_f64_e64 s[0:1], s[0:1], s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cmp_eq_f64_e64 s[0:1], s[0:1], flat_scratch
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_cmp_eq_f64_e64 s[0:1], vcc, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
