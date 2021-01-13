// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx906 %s 2>&1 | FileCheck --check-prefix=GFX906-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx908 %s 2>&1 | FileCheck --check-prefix=GFX908-ERR --implicit-check-not=error: %s

// GFX906-ERR: error: instruction not supported on this GPU
v_dot2c_f32_f16 v0, v1, v2

// GFX906-ERR: error: instruction not supported on this GPU
// GFX908-ERR: error: e64 variant of this instruction is not supported
v_dot2c_f32_f16_e64 v0, v1, v2

// GFX906-ERR: error: instruction not supported on this GPU
// GFX908-ERR: error: sdwa variant of this instruction is not supported
v_dot2c_f32_f16_sdwa v0, v1, v2

// GFX906-ERR: error: instruction not supported on this GPU
v_dot2c_i32_i16 v0, v1, v2

// GFX906-ERR: error: instruction not supported on this GPU
// GFX908-ERR: error: e64 variant of this instruction is not supported
v_dot2c_i32_i16_e64 v0, v1, v2

// GFX906-ERR: error: instruction not supported on this GPU
// GFX908-ERR: error: sdwa variant of this instruction is not supported
v_dot2c_i32_i16_sdwa v0, v1, v2

// GFX906-ERR: error: instruction not supported on this GPU
v_dot4c_i32_i8 v0, v1, v2

// GFX906-ERR: error: instruction not supported on this GPU
// GFX908-ERR: error: e64 variant of this instruction is not supported
v_dot4c_i32_i8_e64 v0, v1, v2

// GFX906-ERR: error: instruction not supported on this GPU
// GFX908-ERR: error: sdwa variant of this instruction is not supported
v_dot4c_i32_i8_sdwa v0, v1, v2

// GFX906-ERR: error: instruction not supported on this GPU
v_dot8c_i32_i4 v0, v1, v2

// GFX906-ERR: error: instruction not supported on this GPU
// GFX908-ERR: error: e64 variant of this instruction is not supported
v_dot8c_i32_i4_e64 v0, v1, v2

// GFX906-ERR: error: instruction not supported on this GPU
// GFX908-ERR: error: sdwa variant of this instruction is not supported
v_dot8c_i32_i4_sdwa v0, v1, v2

// GFX906-ERR: error: instruction not supported on this GPU
v_pk_fmac_f16 v0, v1, v2
