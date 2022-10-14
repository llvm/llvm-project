// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck %s --implicit-check-not=error: --strict-whitespace

//==============================================================================
// 'null' operand is not supported on this GPU

s_add_u32 null, null, null
// CHECK: error: 'null' operand is not supported on this GPU
// CHECK-NEXT:{{^}}s_add_u32 null, null, null
// CHECK-NEXT:{{^}}          ^

//==============================================================================
// ABS not allowed in VOP3B instructions

v_div_scale_f64  v[24:25], vcc, -|v[22:23]|, v[22:23], v[20:21]
// CHECK: error: ABS not allowed in VOP3B instructions
// CHECK-NEXT:{{^}}v_div_scale_f64  v[24:25], vcc, -|v[22:23]|, v[22:23], v[20:21]
// CHECK-NEXT:{{^}}^

//==============================================================================
// dlc modifier is not supported on this GPU

scratch_load_ubyte v1, v2, off dlc
// CHECK: error: dlc modifier is not supported on this GPU
// CHECK-NEXT:{{^}}scratch_load_ubyte v1, v2, off dlc
// CHECK-NEXT:{{^}}                               ^

scratch_load_ubyte v1, v2, off nodlc
// CHECK: error: dlc modifier is not supported on this GPU
// CHECK-NEXT:{{^}}scratch_load_ubyte v1, v2, off nodlc
// CHECK-NEXT:{{^}}                               ^

//==============================================================================
// duplicate VGPR index mode

s_set_gpr_idx_on s0, gpr_idx(SRC0,DST,SRC1,DST)
// CHECK: error: duplicate VGPR index mode
// CHECK-NEXT:{{^}}s_set_gpr_idx_on s0, gpr_idx(SRC0,DST,SRC1,DST)
// CHECK-NEXT:{{^}}                                           ^

//==============================================================================
// exp target is not supported on this GPU

exp pos4 v4, v3, v2, v1
// CHECK: error: exp target is not supported on this GPU
// CHECK-NEXT:{{^}}exp pos4 v4, v3, v2, v1
// CHECK-NEXT:{{^}}    ^

//==============================================================================
// expected a 12-bit unsigned offset

flat_load_dword v1, v[3:4] offset:-1
// CHECK: error: expected a 12-bit unsigned offset
// CHECK-NEXT:{{^}}flat_load_dword v1, v[3:4] offset:-1
// CHECK-NEXT:{{^}}                           ^

flat_load_dword v1, v[3:4] offset:4096
// CHECK: error: expected a 12-bit unsigned offset
// CHECK-NEXT:{{^}}flat_load_dword v1, v[3:4] offset:4096
// CHECK-NEXT:{{^}}                           ^

//==============================================================================
// expected a 13-bit signed offset

global_load_dword v1, v[3:4] off, offset:-4097
// CHECK: error: expected a 13-bit signed offset
// CHECK-NEXT:{{^}}global_load_dword v1, v[3:4] off, offset:-4097
// CHECK-NEXT:{{^}}                                  ^

//==============================================================================
// expected a VGPR index mode

s_set_gpr_idx_on s0, gpr_idx(SRC0,
// CHECK: error: expected a VGPR index mode
// CHECK-NEXT:{{^}}s_set_gpr_idx_on s0, gpr_idx(SRC0,
// CHECK-NEXT:{{^}}                                  ^

//==============================================================================
// expected a VGPR index mode or a closing parenthesis

s_set_gpr_idx_on s0, gpr_idx(
// CHECK: error: expected a VGPR index mode or a closing parenthesis
// CHECK-NEXT:{{^}}s_set_gpr_idx_on s0, gpr_idx(
// CHECK-NEXT:{{^}}                             ^

s_set_gpr_idx_on s0, gpr_idx(X)
// CHECK: error: expected a VGPR index mode or a closing parenthesis
// CHECK-NEXT:{{^}}s_set_gpr_idx_on s0, gpr_idx(X)
// CHECK-NEXT:{{^}}                             ^

//==============================================================================
// expected a comma or a closing parenthesis

s_set_gpr_idx_on s0, gpr_idx(DST
// CHECK: error: expected a comma or a closing parenthesis
// CHECK-NEXT:{{^}}s_set_gpr_idx_on s0, gpr_idx(DST
// CHECK-NEXT:{{^}}                                ^

//==============================================================================
// expected absolute expression

s_set_gpr_idx_on s0, gpr_idx
// CHECK: error: expected absolute expression
// CHECK-NEXT:{{^}}s_set_gpr_idx_on s0, gpr_idx
// CHECK-NEXT:{{^}}                     ^

s_set_gpr_idx_on s0, s1
// CHECK: error: expected absolute expression
// CHECK-NEXT:{{^}}s_set_gpr_idx_on s0, s1
// CHECK-NEXT:{{^}}                     ^

//==============================================================================
// invalid atomic image dmask

image_atomic_add v252, v2, s[8:15]
// CHECK: error: invalid atomic image dmask
// CHECK-NEXT:{{^}}image_atomic_add v252, v2, s[8:15]
// CHECK-NEXT:{{^}}^

image_atomic_cmpswap v[4:7], v[192:195], s[28:35] dmask:0xe tfe
// CHECK: error: invalid atomic image dmask
// CHECK-NEXT:{{^}}image_atomic_cmpswap v[4:7], v[192:195], s[28:35] dmask:0xe tfe
// CHECK-NEXT:{{^}}                                                  ^

//==============================================================================
// invalid image_gather dmask: only one bit must be set

image_gather4_cl v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x3
// CHECK: error: invalid image_gather dmask: only one bit must be set
// CHECK-NEXT:{{^}}image_gather4_cl v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x3
// CHECK-NEXT:{{^}}                                                   ^

//==============================================================================
// invalid immediate: only 4-bit values are legal

s_set_gpr_idx_on s0, 16
// CHECK: error: invalid immediate: only 4-bit values are legal
// CHECK-NEXT:{{^}}s_set_gpr_idx_on s0, 16
// CHECK-NEXT:{{^}}                     ^

//==============================================================================
// invalid operand (violates constant bus restrictions)

v_add_f32_e64 v0, flat_scratch_hi, m0
// CHECK: error: invalid operand (violates constant bus restrictions)
// CHECK-NEXT:{{^}}v_add_f32_e64 v0, flat_scratch_hi, m0
// CHECK-NEXT:{{^}}                                   ^

v_madak_f32 v5, s1, v2, 0xa1b1c1d1
// CHECK: error: invalid operand (violates constant bus restrictions)
// CHECK-NEXT:{{^}}v_madak_f32 v5, s1, v2, 0xa1b1c1d1
// CHECK-NEXT:{{^}}                ^

v_madmk_f32 v5, s1, 0x11213141, v255
// CHECK: error: invalid operand (violates constant bus restrictions)
// CHECK-NEXT:{{^}}v_madmk_f32 v5, s1, 0x11213141, v255
// CHECK-NEXT:{{^}}                ^

//==============================================================================
// literal operands are not supported

v_bfe_u32 v0, v2, v3, undef
// CHECK: error: literal operands are not supported
// CHECK-NEXT:{{^}}v_bfe_u32 v0, v2, v3, undef
// CHECK-NEXT:{{^}}                      ^

v_bfe_u32 v0, v2, undef, v3
// CHECK: error: literal operands are not supported
// CHECK-NEXT:{{^}}v_bfe_u32 v0, v2, undef, v3
// CHECK-NEXT:{{^}}                  ^

v_add_i16 v5, v1, 0.5
// CHECK: error: literal operands are not supported
// CHECK-NEXT:{{^}}v_add_i16 v5, v1, 0.5
// CHECK-NEXT:{{^}}                  ^

v_add_i16 v5, 0.5, v2
// CHECK: error: literal operands are not supported
// CHECK-NEXT:{{^}}v_add_i16 v5, 0.5, v2
// CHECK-NEXT:{{^}}              ^

//==============================================================================
// missing dst operand or lds modifier

buffer_load_dword off, s[8:11], s3
// CHECK: error: missing dst operand or lds modifier
// CHECK-NEXT:{{^}}buffer_load_dword off, s[8:11], s3
// CHECK-NEXT:{{^}}^

buffer_load_dword off, s[8:11], s3 offset:1
// CHECK: error: missing dst operand or lds modifier
// CHECK-NEXT:{{^}}buffer_load_dword off, s[8:11], s3 offset:1
// CHECK-NEXT:{{^}}^

//==============================================================================
// r128 modifier is not supported on this GPU

image_atomic_add v10, v6, s[8:15] dmask:0x1 r128
// CHECK: error: r128 modifier is not supported on this GPU
// CHECK-NEXT:{{^}}image_atomic_add v10, v6, s[8:15] dmask:0x1 r128
// CHECK-NEXT:{{^}}                                            ^

image_atomic_add v10, v6, s[8:15] dmask:0x1 nor128
// CHECK: error: r128 modifier is not supported on this GPU
// CHECK-NEXT:{{^}}image_atomic_add v10, v6, s[8:15] dmask:0x1 nor128
// CHECK-NEXT:{{^}}                                            ^

//==============================================================================
// unified format is not supported on this GPU

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_SNORM,BUF_DATA_FORMAT_8] idxen
// CHECK: error: unified format is not supported on this GPU
// CHECK-NEXT:{{^}}tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_SNORM,BUF_DATA_FORMAT_8] idxen
// CHECK-NEXT:{{^}}                                                         ^

//==============================================================================
// duplicate format

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:1 s0 format:[BUF_NUM_FORMAT_FLOAT]
// CHECK: error: duplicate format
// CHECK-NEXT:{{^}}tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:1 s0 format:[BUF_NUM_FORMAT_FLOAT]
// CHECK-NEXT:{{^}}                                                            ^

//==============================================================================
// not a valid operand

v_cndmask_b32_sdwa v5, sext(v1), v2, vcc dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:BYTE_0 src1_sel:WORD_0
// CHECK: error: not a valid operand.
// CHECK-NEXT:{{^}}v_cndmask_b32_sdwa v5, sext(v1), v2, vcc dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:BYTE_0 src1_sel:WORD_0
// CHECK-NEXT:{{^}}                       ^

//==============================================================================
// out of range dfmt

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:16 nfmt:1 s0
// CHECK: error: out of range dfmt
// CHECK-NEXT:{{^}}tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:16 nfmt:1 s0
// CHECK-NEXT:{{^}}                                                 ^

//==============================================================================
// out of range nfmt

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:1 nfmt:-1 s0
// CHECK: error: out of range nfmt
// CHECK-NEXT:{{^}}tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:1 nfmt:-1 s0
// CHECK-NEXT:{{^}}                                                        ^
