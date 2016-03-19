// RUN: llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=CIVI --check-prefix=VI
// RUN: not llvm-mc -arch=amdgcn -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOSI --check-prefix=NOSICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=SI -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOSI --check-prefix=NOSICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOSICI

//===----------------------------------------------------------------------===//
// Check dpp_ctrl values
//===----------------------------------------------------------------------===//

// NOSICI: error:
// VI: v_mov_b32_dpp v0, v0 quad_perm:[0,2,1,1] row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x58,0x00,0xff]
v_mov_b32 v0, v0 quad_perm:[0,2,1,1]

// NOSICI: error:
// VI: v_mov_b32_dpp v0, v0 row_shl:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x01,0x01,0xff]
v_mov_b32 v0, v0 row_shl:1

// NOSICI: error:
// VI: v_mov_b32_dpp v0, v0 row_shr:15 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x1f,0x01,0xff]
v_mov_b32 v0, v0 row_shr:0xf

// NOSICI: error:
// VI: v_mov_b32_dpp v0, v0 row_ror:12 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x2c,0x01,0xff]
v_mov_b32 v0, v0 row_ror:0xc

// NOSICI: error:
// VI: v_mov_b32_dpp v0, v0 wave_shl:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x30,0x01,0xff]
v_mov_b32 v0, v0 wave_shl:1

// NOSICI: error:
// VI: v_mov_b32_dpp v0, v0 wave_rol:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x34,0x01,0xff]
v_mov_b32 v0, v0 wave_rol:1

// NOSICI: error:
// VI: v_mov_b32_dpp v0, v0 wave_shr:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x38,0x01,0xff]
v_mov_b32 v0, v0 wave_shr:1

// NOSICI: error:
// VI: v_mov_b32_dpp v0, v0 wave_ror:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x3c,0x01,0xff]
v_mov_b32 v0, v0 wave_ror:1

// NOSICI: error:
// VI: v_mov_b32_dpp v0, v0 row_mirror row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x40,0x01,0xff]
v_mov_b32 v0, v0 row_mirror

// NOSICI: error:
// VI: v_mov_b32_dpp v0, v0 row_half_mirror row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x41,0x01,0xff]
v_mov_b32 v0, v0 row_half_mirror

// NOSICI: error:
// VI: v_mov_b32_dpp v0, v0 row_bcast:15 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x42,0x01,0xff]
v_mov_b32 v0, v0 row_bcast:15

// NOSICI: error:
// VI: v_mov_b32_dpp v0, v0 row_bcast:31 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x43,0x01,0xff]
v_mov_b32 v0, v0 row_bcast:31

//===----------------------------------------------------------------------===//
// Check optional fields
//===----------------------------------------------------------------------===//

// NOSICI: error:
// VI: v_mov_b32_dpp v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bank_mask:0x1 bound_ctrl:0 ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x4d,0x08,0xa1]
v_mov_b32 v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error:
// VI: v_mov_b32_dpp v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bank_mask:0xf ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x4d,0x00,0xaf]
v_mov_b32 v0, v0 quad_perm:[1,3,0,1] row_mask:0xa

// NOSICI: error:
// VI: v_mov_b32_dpp v0, v0 quad_perm:[1,3,0,1] row_mask:0xf bank_mask:0x1 ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x4d,0x00,0xf1]
v_mov_b32 v0, v0 quad_perm:[1,3,0,1] bank_mask:0x1

// NOSICI: error:
// VI: v_mov_b32_dpp v0, v0 quad_perm:[1,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:0 ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x4d,0x08,0xff]
v_mov_b32 v0, v0 quad_perm:[1,3,0,1] bound_ctrl:0

// NOSICI: error:
// VI: v_mov_b32_dpp v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bank_mask:0x1 ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x4d,0x00,0xa1]
v_mov_b32 v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bank_mask:0x1

// NOSICI: error:
// VI: v_mov_b32_dpp v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bank_mask:0xf bound_ctrl:0 ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x4d,0x08,0xaf]
v_mov_b32 v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bound_ctrl:0

// NOSICI: error:
// VI: v_mov_b32_dpp v0, v0 quad_perm:[1,3,0,1] row_mask:0xf bank_mask:0x1 bound_ctrl:0 ; encoding: [0xfa,0x02,0x00,0x7e,0x00,0x4d,0x08,0xf1]
v_mov_b32 v0, v0 quad_perm:[1,3,0,1] bank_mask:0x1 bound_ctrl:0

//===----------------------------------------------------------------------===//
// Check VOP1 opcodes
//===----------------------------------------------------------------------===//
// ToDo: v_nop

// NOSICI: error:
// VI: v_cvt_u32_f32_dpp v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0 ; encoding: [0xfa,0x0e,0x00,0x7e,0x00,0x01,0x09,0xa1]
v_cvt_u32_f32 v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error:
// VI: v_fract_f32_dpp v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0 ; encoding: [0xfa,0x36,0x00,0x7e,0x00,0x01,0x09,0xa1]
v_fract_f32 v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error:
// VI: v_sin_f32_dpp v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0 ; encoding: [0xfa,0x52,0x00,0x7e,0x00,0x01,0x09,0xa1]
v_sin_f32 v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

//===----------------------------------------------------------------------===//
// Check VOP2 opcodes
//===----------------------------------------------------------------------===//
// ToDo: VOP2bInst instructions: v_add_u32, v_sub_u32 ... (vcc and ApplyMnemonic in AsmMatcherEmitter.cpp)
// ToDo: v_mac_f32 (VOP_MAC)

// NOSICI: error:
// VI: v_add_f32_dpp v0, v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0 ; encoding: [0xfa,0x00,0x00,0x02,0x00,0x01,0x09,0xa1]
v_add_f32 v0, v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error:
// VI: v_min_f32_dpp v0, v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0 ; encoding: [0xfa,0x00,0x00,0x14,0x00,0x01,0x09,0xa1]
v_min_f32 v0, v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error:
// VI: v_and_b32_dpp v0, v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0 ; encoding: [0xfa,0x00,0x00,0x26,0x00,0x01,0x09,0xa1]
v_and_b32 v0, v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

//===----------------------------------------------------------------------===//
// Check modifiers
//===----------------------------------------------------------------------===//

// NOSICI: error:
// VI: v_add_f32_dpp v0, -v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0 ; encoding: [0xfa,0x00,0x00,0x02,0x00,0x01,0x19,0xa1]
v_add_f32 v0, -v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error:
// VI: v_add_f32_dpp v0, v0, |v0| row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0 ; encoding: [0xfa,0x00,0x00,0x02,0x00,0x01,0x89,0xa1]
v_add_f32 v0, v0, |v0| row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error:
// VI: v_add_f32_dpp v0, -v0, |v0| row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0 ; encoding: [0xfa,0x00,0x00,0x02,0x00,0x01,0x99,0xa1]
v_add_f32 v0, -v0, |v0| row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

// NOSICI: error:
// VI: v_add_f32_dpp v0, |v0|, -v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0 ; encoding: [0xfa,0x00,0x00,0x02,0x00,0x01,0x69,0xa1]
v_add_f32 v0, |v0|, -v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0
