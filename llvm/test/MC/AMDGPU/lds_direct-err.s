// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck %s --check-prefix=NOGFX9 --implicit-check-not=error:

//---------------------------------------------------------------------------//
// lds_direct may be used only with vector ALU instructions
//---------------------------------------------------------------------------//

s_and_b32 s2, lds_direct, s1
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

//---------------------------------------------------------------------------//
// lds_direct may not be used with "REV" opcodes
//---------------------------------------------------------------------------//

v_ashrrev_i16 v0, lds_direct, v0
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct cannot be used with this instruction

v_ashrrev_i32 v0, lds_direct, v0
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct cannot be used with this instruction

v_lshlrev_b16 v0, lds_direct, v0
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct cannot be used with this instruction

v_lshlrev_b32 v0, lds_direct, v0
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct cannot be used with this instruction

v_lshrrev_b16 v0, lds_direct, v0
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct cannot be used with this instruction

v_lshrrev_b32 v0, lds_direct, v0
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct cannot be used with this instruction

v_pk_ashrrev_i16 v0, lds_direct, v0
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct cannot be used with this instruction

v_pk_lshlrev_b16 v0, lds_direct, v0
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct cannot be used with this instruction

v_pk_lshrrev_b16 v0, lds_direct, v0
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct cannot be used with this instruction

v_subbrev_co_u32 v0, vcc, src_lds_direct, v0, vcc
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct cannot be used with this instruction

v_subrev_co_u32 v0, vcc, src_lds_direct, v0
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct cannot be used with this instruction

v_subrev_f16 v0, src_lds_direct, v0
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct cannot be used with this instruction

v_subrev_u16 v0, src_lds_direct, v0
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct cannot be used with this instruction

v_subrev_u32 v0, src_lds_direct, v0
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct cannot be used with this instruction

//---------------------------------------------------------------------------//
// lds_direct may not be used with v_writelane_b32 for VI/GFX9
//---------------------------------------------------------------------------//

v_writelane_b32 v0, lds_direct, s0
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

//---------------------------------------------------------------------------//
// lds_direct cannot be used with 64-bit and larger operands
//---------------------------------------------------------------------------//

v_add_f64 v[0:1], lds_direct, v[0:1]
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

//---------------------------------------------------------------------------//
// Only SRC0 may specify lds_direct
//---------------------------------------------------------------------------//

v_add_i32 v0, v0, lds_direct
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct may be used as src0 only

v_add_i32 lds_direct, v0, v0
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_fma_f32 v0, v0, v0, lds_direct
// NOGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct may be used as src0 only
