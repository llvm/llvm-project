// RUN: llvm-mc -arch=amdgcn -mcpu=gfx940 -show-encoding %s | FileCheck --check-prefix=GFX940 --strict-whitespace %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx90a %s 2>&1 | FileCheck --check-prefixes=NOT-GFX940,GFX90A --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck --check-prefixes=NOT-GFX940,GFX10 --implicit-check-not=error: %s

// NOT-GFX940: error: invalid operand for instruction
// GFX940: global_load_dword v2, v[2:3], off sc0   ; encoding: [0x00,0x80,0x51,0xdc,0x02,0x00,0x7f,0x02]
global_load_dword v2, v[2:3], off sc0

// NOT-GFX940: error: invalid operand for instruction
// GFX940: global_load_dword v2, v[2:3], off       ; encoding: [0x00,0x80,0x50,0xdc,0x02,0x00,0x7f,0x02]
global_load_dword v2, v[2:3], off nosc0

// NOT-GFX940: error: invalid operand for instruction
// GFX940: global_load_dword v2, v[2:3], off sc1   ; encoding: [0x00,0x80,0x50,0xde,0x02,0x00,0x7f,0x02]
global_load_dword v2, v[2:3], off sc1

// NOT-GFX940: error: invalid operand for instruction
// GFX940: global_load_dword v2, v[2:3], off       ; encoding: [0x00,0x80,0x50,0xdc,0x02,0x00,0x7f,0x02]
global_load_dword v2, v[2:3], off nosc1

// NOT-GFX940: error: invalid operand for instruction
// GFX940: global_load_dword v2, v[2:3], off nt    ; encoding: [0x00,0x80,0x52,0xdc,0x02,0x00,0x7f,0x02]
global_load_dword v2, v[2:3], off nt

// NOT-GFX940: error: invalid operand for instruction
// GFX940: global_load_dword v2, v[2:3], off       ; encoding: [0x00,0x80,0x50,0xdc,0x02,0x00,0x7f,0x02]
global_load_dword v2, v[2:3], off nont

// GFX940: s_load_dword s2, s[2:3], 0x0 glc        ; encoding: [0x81,0x00,0x03,0xc0,0x00,0x00,0x00,0x00]
s_load_dword s2, s[2:3], 0x0 glc

// NOT-GFX940: error: invalid operand for instruction
// GFX940: buffer_load_dword v5, off, s[8:11], s3 sc0 nt sc1 ; encoding: [0x00,0xc0,0x52,0xe0,0x00,0x05,0x02,0x03]
buffer_load_dword v5, off, s[8:11], s3 sc0 nt sc1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: flat_atomic_add_f32 v[2:3], v1          ; encoding: [0x00,0x00,0x34,0xdd,0x02,0x01,0x00,0x00]
flat_atomic_add_f32 v[2:3], v1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: flat_atomic_add_f32 v[2:3], a1          ; encoding: [0x00,0x00,0x34,0xdd,0x02,0x01,0x80,0x00]
flat_atomic_add_f32 v[2:3], a1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: flat_atomic_add_f32 v4, v[2:3], v1 sc0  ; encoding: [0x00,0x00,0x35,0xdd,0x02,0x01,0x00,0x04]
flat_atomic_add_f32 v4, v[2:3], v1 sc0

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: flat_atomic_add_f32 a4, v[2:3], a1 sc0  ; encoding: [0x00,0x00,0x35,0xdd,0x02,0x01,0x80,0x04]
flat_atomic_add_f32 a4, v[2:3], a1 sc0

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: flat_atomic_pk_add_f16 v4, v[2:3], v1 sc0 ; encoding: [0x00,0x00,0x39,0xdd,0x02,0x01,0x00,0x04]
flat_atomic_pk_add_f16 v4, v[2:3], v1 sc0

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: flat_atomic_pk_add_f16 a4, v[2:3], a1 sc0 ; encoding: [0x00,0x00,0x39,0xdd,0x02,0x01,0x80,0x04]
flat_atomic_pk_add_f16 a4, v[2:3], a1 sc0

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: flat_atomic_pk_add_f16 v[2:3], v1       ; encoding: [0x00,0x00,0x38,0xdd,0x02,0x01,0x00,0x00]
flat_atomic_pk_add_f16 v[2:3], v1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: flat_atomic_pk_add_f16 v[2:3], a1       ; encoding: [0x00,0x00,0x38,0xdd,0x02,0x01,0x80,0x00]
flat_atomic_pk_add_f16 v[2:3], a1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: flat_atomic_pk_add_bf16 v4, v[2:3], v1 sc0 ; encoding: [0x00,0x00,0x49,0xdd,0x02,0x01,0x00,0x04]
flat_atomic_pk_add_bf16 v4, v[2:3], v1 sc0

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: flat_atomic_pk_add_bf16 a4, v[2:3], a1 sc0 ; encoding: [0x00,0x00,0x49,0xdd,0x02,0x01,0x80,0x04]
flat_atomic_pk_add_bf16 a4, v[2:3], a1 sc0

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: flat_atomic_pk_add_bf16 v[2:3], v1      ; encoding: [0x00,0x00,0x48,0xdd,0x02,0x01,0x00,0x00]
flat_atomic_pk_add_bf16 v[2:3], v1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: flat_atomic_pk_add_bf16 v[2:3], a1      ; encoding: [0x00,0x00,0x48,0xdd,0x02,0x01,0x80,0x00]
flat_atomic_pk_add_bf16 v[2:3], a1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: global_atomic_pk_add_bf16 v4, v[2:3], v1, off sc0 ; encoding: [0x00,0x80,0x49,0xdd,0x02,0x01,0x7f,0x04]
global_atomic_pk_add_bf16 v4, v[2:3], v1, off sc0

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: global_atomic_pk_add_bf16 a4, v[2:3], a1, off sc0 ; encoding: [0x00,0x80,0x49,0xdd,0x02,0x01,0xff,0x04]
global_atomic_pk_add_bf16 a4, v[2:3], a1, off sc0

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: global_atomic_pk_add_bf16 v[2:3], v1, off ; encoding: [0x00,0x80,0x48,0xdd,0x02,0x01,0x7f,0x00]
global_atomic_pk_add_bf16 v[2:3], v1, off

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: global_atomic_pk_add_bf16 v[2:3], a1, off ; encoding: [0x00,0x80,0x48,0xdd,0x02,0x01,0xff,0x00]
global_atomic_pk_add_bf16 v[2:3], a1, off

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: ds_pk_add_f16 v2, v1                    ; encoding: [0x00,0x00,0x2e,0xd8,0x02,0x01,0x00,0x00]
ds_pk_add_f16 v2, v1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: ds_pk_add_f16 v2, a1                    ; encoding: [0x00,0x00,0x2e,0xda,0x02,0x01,0x00,0x00]
ds_pk_add_f16 v2, a1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: ds_pk_add_rtn_f16 v3, v2, v1            ; encoding: [0x00,0x00,0x6e,0xd9,0x02,0x01,0x00,0x03]
ds_pk_add_rtn_f16  v3, v2, v1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: ds_pk_add_rtn_f16 a3, v2, a1            ; encoding: [0x00,0x00,0x6e,0xdb,0x02,0x01,0x00,0x03]
ds_pk_add_rtn_f16  a3, v2, a1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: ds_pk_add_bf16 v2, v1                   ; encoding: [0x00,0x00,0x30,0xd8,0x02,0x01,0x00,0x00]
ds_pk_add_bf16 v2, v1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: ds_pk_add_bf16 v2, a1                   ; encoding: [0x00,0x00,0x30,0xda,0x02,0x01,0x00,0x00]
ds_pk_add_bf16 v2, a1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: ds_pk_add_rtn_bf16 v3, v2, v1           ; encoding: [0x00,0x00,0x70,0xd9,0x02,0x01,0x00,0x03]
ds_pk_add_rtn_bf16  v3, v2, v1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: ds_pk_add_rtn_bf16 a3, v2, a1           ; encoding: [0x00,0x00,0x70,0xdb,0x02,0x01,0x00,0x03]
ds_pk_add_rtn_bf16  a3, v2, a1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: global_load_lds_dword v[2:3], off       ; encoding: [0x00,0x80,0xa8,0xdc,0x02,0x00,0x7f,0x00]
global_load_lds_dword v[2:3], off

// NOT-GFX940: error:
// GFX940: global_load_lds_dword v[2:3], off sc0 nt sc1 ; encoding: [0x00,0x80,0xab,0xde,0x02,0x00,0x7f,0x00]
global_load_lds_dword v[2:3], off sc0 nt sc1

// NOT-GFX940: error:
// GFX940: global_load_lds_dword v[2:3], off offset:4 ; encoding: [0x04,0x80,0xa8,0xdc,0x02,0x00,0x7f,0x00]
global_load_lds_dword v[2:3], off offset:4

// NOT-GFX940: error:
// GFX940: global_load_lds_dword v2, s[4:5] offset:4 ; encoding: [0x04,0x80,0xa8,0xdc,0x02,0x00,0x04,0x00]
global_load_lds_dword v2, s[4:5] offset:4

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: global_load_lds_ubyte v[2:3], off       ; encoding: [0x00,0x80,0x98,0xdc,0x02,0x00,0x7f,0x00]
global_load_lds_ubyte v[2:3], off

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: global_load_lds_sbyte v[2:3], off       ; encoding: [0x00,0x80,0x9c,0xdc,0x02,0x00,0x7f,0x00]
global_load_lds_sbyte v[2:3], off

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: global_load_lds_sshort v[2:3], off      ; encoding: [0x00,0x80,0xa4,0xdc,0x02,0x00,0x7f,0x00]
global_load_lds_sshort v[2:3], off

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: global_load_lds_ushort v[2:3], off      ; encoding: [0x00,0x80,0xa0,0xdc,0x02,0x00,0x7f,0x00]
global_load_lds_ushort v[2:3], off

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: scratch_load_lds_dword v2, off          ; encoding: [0x00,0x60,0xa8,0xdc,0x02,0x00,0x7f,0x00]
scratch_load_lds_dword v2, off

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: scratch_load_lds_dword v2, s4           ; encoding: [0x00,0x60,0xa8,0xdc,0x02,0x00,0x04,0x00]
scratch_load_lds_dword v2, s4

// NOT-GFX940: error:
// GFX940: scratch_load_lds_dword v2, s4 offset:4  ; encoding: [0x04,0x60,0xa8,0xdc,0x02,0x00,0x04,0x00]
scratch_load_lds_dword v2, s4 offset:4

// NOT-GFX940: error:
// GFX940: scratch_load_lds_dword off, s4 offset:4 ; encoding: [0x04,0x40,0xa8,0xdc,0x00,0x00,0x04,0x00]
scratch_load_lds_dword off, s4 offset:4

// NOT-GFX940: error:
// GFX940: scratch_load_lds_dword off, off offset:4 ; encoding: [0x04,0x40,0xa8,0xdc,0x00,0x00,0x7f,0x00]
scratch_load_lds_dword off, off offset:4

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: scratch_load_lds_ubyte v2, off          ; encoding: [0x00,0x60,0x98,0xdc,0x02,0x00,0x7f,0x00]
scratch_load_lds_ubyte v2, off

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: scratch_load_lds_sbyte v2, off          ; encoding: [0x00,0x60,0x9c,0xdc,0x02,0x00,0x7f,0x00]
scratch_load_lds_sbyte v2, off

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: scratch_load_lds_ushort v2, off         ; encoding: [0x00,0x60,0xa0,0xdc,0x02,0x00,0x7f,0x00]
scratch_load_lds_ushort v2, off

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: scratch_load_lds_sshort v2, off         ; encoding: [0x00,0x60,0xa4,0xdc,0x02,0x00,0x7f,0x00]
scratch_load_lds_sshort v2, off

// NOT-GFX940: error: specified hardware register is not supported on this GPU
// GFX940: s_getreg_b32 s1, hwreg(HW_REG_XCC_ID)   ; encoding: [0x14,0xf8,0x81,0xb8]
s_getreg_b32 s1, hwreg(HW_REG_XCC_ID)

// NOT-GFX940: error: specified hardware register is not supported on this GPU
// GFX940: s_getreg_b32 s1, hwreg(HW_REG_SQ_PERF_SNAPSHOT_DATA) ; encoding: [0x15,0xf8,0x81,0xb8]
s_getreg_b32 s1, hwreg(HW_REG_SQ_PERF_SNAPSHOT_DATA)

// NOT-GFX940: error: specified hardware register is not supported on this GPU
// GFX940: s_getreg_b32 s1, hwreg(HW_REG_SQ_PERF_SNAPSHOT_DATA1) ; encoding: [0x16,0xf8,0x81,0xb8]
s_getreg_b32 s1, hwreg(HW_REG_SQ_PERF_SNAPSHOT_DATA1)

// NOT-GFX940: error: specified hardware register is not supported on this GPU
// GFX940: s_getreg_b32 s1, hwreg(HW_REG_SQ_PERF_SNAPSHOT_PC_LO) ; encoding: [0x17,0xf8,0x81,0xb8]
s_getreg_b32 s1, hwreg(HW_REG_SQ_PERF_SNAPSHOT_PC_LO)

// NOT-GFX940: error: specified hardware register is not supported on this GPU
// GFX940: s_getreg_b32 s1, hwreg(HW_REG_SQ_PERF_SNAPSHOT_PC_HI) ; encoding: [0x18,0xf8,0x81,0xb8]
s_getreg_b32 s1, hwreg(HW_REG_SQ_PERF_SNAPSHOT_PC_HI)

// GFX940: s_getreg_b32 s1, hwreg(HW_REG_TMA_HI)   ; encoding: [0x13,0xf8,0x81,0xb8]
s_getreg_b32 s1, hwreg(19)

// GFX940: s_getreg_b32 s1, hwreg(HW_REG_XCC_ID)   ; encoding: [0x14,0xf8,0x81,0xb8]
s_getreg_b32 s1, hwreg(20)

// GFX940: s_getreg_b32 s1, hwreg(HW_REG_SQ_PERF_SNAPSHOT_DATA) ; encoding: [0x15,0xf8,0x81,0xb8]
s_getreg_b32 s1, hwreg(21)

// GFX940: s_getreg_b32 s1, hwreg(HW_REG_SQ_PERF_SNAPSHOT_DATA1) ; encoding: [0x16,0xf8,0x81,0xb8]
s_getreg_b32 s1, hwreg(22)

// GFX940: s_getreg_b32 s1, hwreg(HW_REG_SQ_PERF_SNAPSHOT_PC_LO) ; encoding: [0x17,0xf8,0x81,0xb8]
s_getreg_b32 s1, hwreg(23)

// GFX940: s_getreg_b32 s1, hwreg(HW_REG_SQ_PERF_SNAPSHOT_PC_HI) ; encoding: [0x18,0xf8,0x81,0xb8]
s_getreg_b32 s1, hwreg(24)

// GFX940: s_getreg_b32 s1, hwreg(25)              ; encoding: [0x19,0xf8,0x81,0xb8]
s_getreg_b32 s1, hwreg(25)

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: v_mov_b64_e32 v[2:3], v[4:5]            ; encoding: [0x04,0x71,0x04,0x7e]
v_mov_b64 v[2:3], v[4:5]

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: v_mov_b64_dpp v[2:3], v[4:5] row_newbcast:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x70,0x04,0x7e,0x04,0x51,0x01,0xff]
v_mov_b64 v[2:3], v[4:5]  row_newbcast:1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: v_mov_b64_e32 v[2:3], s[4:5]            ; encoding: [0x04,0x70,0x04,0x7e]
v_mov_b64 v[2:3], s[4:5]

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: v_mov_b64_e32 v[2:3], 1                 ; encoding: [0x81,0x70,0x04,0x7e]
v_mov_b64 v[2:3], 1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: v_mov_b64_e32 v[2:3], 0x64              ; encoding: [0xff,0x70,0x04,0x7e,0x64,0x00,0x00,0x00]
v_mov_b64 v[2:3], 0x64

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: v_lshl_add_u64 v[2:3], s[4:5], v7, v[8:9] ; encoding: [0x02,0x00,0x08,0xd2,0x04,0x0e,0x22,0x04]
v_lshl_add_u64 v[2:3], s[4:5], v7, v[8:9]

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: v_lshl_add_u64 v[2:3], v[4:5], 0, 1     ; encoding: [0x02,0x00,0x08,0xd2,0x04,0x01,0x05,0x02]
v_lshl_add_u64 v[2:3], v[4:5], 0, 1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: v_lshl_add_u64 v[2:3], v[4:5], 3, s[2:3] ; encoding: [0x02,0x00,0x08,0xd2,0x04,0x07,0x09,0x00]
v_lshl_add_u64 v[2:3], v[4:5], 3, s[2:3]

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: v_lshl_add_u64 v[2:3], s[4:5], 4, v[2:3] ; encoding: [0x02,0x00,0x08,0xd2,0x04,0x08,0x09,0x04]
v_lshl_add_u64 v[2:3], s[4:5], 4, v[2:3]

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940: buffer_wbl2 sc1                         ; encoding: [0x00,0x80,0xa0,0xe0,0x00,0x00,0x00,0x00]
buffer_wbl2 sc1

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940: buffer_wbl2 sc0                         ; encoding: [0x00,0x40,0xa0,0xe0,0x00,0x00,0x00,0x00]
buffer_wbl2 sc0

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940: buffer_wbl2 sc0 sc1                     ; encoding: [0x00,0xc0,0xa0,0xe0,0x00,0x00,0x00,0x00]
buffer_wbl2 sc0 sc1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: buffer_inv sc0                          ; encoding: [0x00,0x40,0xa4,0xe0,0x00,0x00,0x00,0x00]
buffer_inv sc0

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: buffer_inv sc1                          ; encoding: [0x00,0x80,0xa4,0xe0,0x00,0x00,0x00,0x00]
buffer_inv sc1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: buffer_inv sc0 sc1                      ; encoding: [0x00,0xc0,0xa4,0xe0,0x00,0x00,0x00,0x00]
buffer_inv sc0 sc1

// NOT-GFX940: error: invalid operand for instruction
// GFX940: buffer_atomic_swap v5, off, s[8:11], s3 sc0 ; encoding: [0x00,0x40,0x00,0xe1,0x00,0x05,0x02,0x03]
buffer_atomic_swap v5, off, s[8:11], s3 sc0

// NOT-GFX940: error: invalid operand for instruction
// GFX940: buffer_atomic_swap v5, off, s[8:11], s3 nt ; encoding: [0x00,0x00,0x02,0xe1,0x00,0x05,0x02,0x03]
buffer_atomic_swap v5, off, s[8:11], s3 nt

// GFX90A: error: instruction not supported on this GPU
// GFX940: v_fmamk_f32 v0, v2, 0x42c80000, v3      ; encoding: [0x02,0x07,0x00,0x2e,0x00,0x00,0xc8,0x42]
v_fmamk_f32 v0, v2, 100.0, v3

// GFX90A: error: instruction not supported on this GPU
// GFX940: v_fmaak_f32 v0, v2, v3, 0x42c80000      ; encoding: [0x02,0x07,0x00,0x30,0x00,0x00,0xc8,0x42]
v_fmaak_f32 v0, v2, v3, 100.0

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940: global_atomic_add_f32 v0, v[0:1], v2, off sc0 sc1 ; encoding: [0x00,0x80,0x35,0xdf,0x00,0x02,0x7f,0x00]
global_atomic_add_f32 v0, v[0:1], v2, off sc0 sc1

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940:  global_atomic_add_f32 v[0:1], v2, off sc1 ; encoding: [0x00,0x80,0x34,0xdf,0x00,0x02,0x7f,0x00]
global_atomic_add_f32 v[0:1], v2, off sc1

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940: global_atomic_add_f32 v0, v2, s[0:1] sc1 ; encoding: [0x00,0x80,0x34,0xdf,0x00,0x02,0x00,0x00]
global_atomic_add_f32 v0, v2, s[0:1] sc1

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940: global_atomic_add_f32 v1, v0, v2, s[0:1] sc0 sc1 ; encoding: [0x00,0x80,0x35,0xdf,0x00,0x02,0x00,0x01]
global_atomic_add_f32 v1, v0, v2, s[0:1] sc0 sc1

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940: global_atomic_pk_add_f16 v0, v[0:1], v2, off sc0 sc1 ; encoding: [0x00,0x80,0x39,0xdf,0x00,0x02,0x7f,0x00]
global_atomic_pk_add_f16 v0, v[0:1], v2, off sc0 sc1

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940: flat_atomic_add_f64 v[0:1], v[0:1], v[2:3] sc0 sc1 ; encoding: [0x00,0x00,0x3d,0xdf,0x00,0x02,0x00,0x00]
flat_atomic_add_f64 v[0:1], v[0:1], v[2:3] sc0 sc1

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940: flat_atomic_add_f64 v[0:1], v[2:3] sc1  ; encoding: [0x00,0x00,0x3c,0xdf,0x00,0x02,0x00,0x00]
flat_atomic_add_f64 v[0:1], v[2:3] sc1

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940: flat_atomic_min_f64 v[0:1], v[2:3] sc1  ; encoding: [0x00,0x00,0x40,0xdf,0x00,0x02,0x00,0x00]
flat_atomic_min_f64 v[0:1], v[2:3] sc1

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940: flat_atomic_max_f64 v[0:1], v[2:3] sc1  ; encoding: [0x00,0x00,0x44,0xdf,0x00,0x02,0x00,0x00]
flat_atomic_max_f64 v[0:1], v[2:3] sc1

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940: global_atomic_add_f64 v[0:1], v[2:3], off sc1 ; encoding: [0x00,0x80,0x3c,0xdf,0x00,0x02,0x7f,0x00]
global_atomic_add_f64 v[0:1], v[2:3], off sc1

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940: global_atomic_min_f64 v[0:1], v[2:3], off sc1 ; encoding: [0x00,0x80,0x40,0xdf,0x00,0x02,0x7f,0x00]
global_atomic_min_f64 v[0:1], v[2:3], off sc1

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940: global_atomic_max_f64 v[0:1], v[2:3], off sc1 ; encoding: [0x00,0x80,0x44,0xdf,0x00,0x02,0x7f,0x00]
global_atomic_max_f64 v[0:1], v[2:3], off sc1

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940: buffer_atomic_add_f32 v4, off, s[8:11], s3 sc1 ; encoding: [0x00,0x80,0x34,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_add_f32 v4, off, s[8:11], s3 sc1

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940: buffer_atomic_pk_add_f16 v4, off, s[8:11], s3 sc1 ; encoding: [0x00,0x80,0x38,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_pk_add_f16 v4, off, s[8:11], s3 sc1

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940: buffer_atomic_add_f64 v[4:5], off, s[8:11], s3 sc1 ; encoding: [0x00,0x80,0x3c,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_add_f64 v[4:5], off, s[8:11], s3 sc1

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940: buffer_atomic_max_f64 v[4:5], off, s[8:11], s3 sc1 ; encoding: [0x00,0x80,0x44,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_max_f64 v[4:5], off, s[8:11], s3 sc1

// GFX90A: error: invalid operand for instruction
// GFX10:  error: instruction not supported on this GPU
// GFX940: buffer_atomic_min_f64 v[4:5], off, s[8:11], s3 sc1 ; encoding: [0x00,0x80,0x40,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_min_f64 v[4:5], off, s[8:11], s3 sc1
