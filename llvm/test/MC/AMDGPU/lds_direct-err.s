// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck %s --check-prefix=NOGFX9

//---------------------------------------------------------------------------//
// lds_direct may be used only with vector ALU instructions
//---------------------------------------------------------------------------//

s_and_b32 s2, lds_direct, s1
// NOGFX9: error

//---------------------------------------------------------------------------//
// lds_direct may not be used with V_{LSHL,LSHR,ASHL}REV opcodes
//---------------------------------------------------------------------------//

v_ashrrev_i16 v0, lds_direct, v0
// NOGFX9: error

v_ashrrev_i32 v0, lds_direct, v0
// NOGFX9: error

v_lshlrev_b16 v0, lds_direct, v0
// NOGFX9: error

v_lshlrev_b32 v0, lds_direct, v0
// NOGFX9: error

v_lshrrev_b16 v0, lds_direct, v0
// NOGFX9: error

v_lshrrev_b32 v0, lds_direct, v0
// NOGFX9: error

v_pk_ashrrev_i16 v0, lds_direct, v0
// NOGFX9: error

v_pk_lshlrev_b16 v0, lds_direct, v0
// NOGFX9: error

v_pk_lshrrev_b16 v0, lds_direct, v0
// NOGFX9: error

//---------------------------------------------------------------------------//
// lds_direct cannot be used with 64-bit and larger operands
//---------------------------------------------------------------------------//

v_add_f64 v[0:1], lds_direct, v[0:1]
// NOGFX9: error

//---------------------------------------------------------------------------//
// Only SRC0 may specify lds_direct
//---------------------------------------------------------------------------//

v_add_i32 v0, v0, lds_direct
// NOGFX9: error

v_add_i32 lds_direct, v0, v0
// NOGFX9: error

v_fma_f32 v0, v0, v0, lds_direct
// NOGFX9: error
