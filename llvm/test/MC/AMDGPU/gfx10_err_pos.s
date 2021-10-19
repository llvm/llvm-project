// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=+WavefrontSize32,-WavefrontSize64 %s 2>&1 | FileCheck %s --implicit-check-not=error: --strict-whitespace

//==============================================================================
// destination must be different than all sources

v_mqsad_pk_u16_u8 v[0:1], v[1:2], v9, v[4:5]
// CHECK: error: destination must be different than all sources
// CHECK-NEXT:{{^}}v_mqsad_pk_u16_u8 v[0:1], v[1:2], v9, v[4:5]
// CHECK-NEXT:{{^}}                          ^

v_mqsad_pk_u16_u8 v[0:1], v[2:3], v0, v[4:5]
// CHECK: error: destination must be different than all sources
// CHECK-NEXT:{{^}}v_mqsad_pk_u16_u8 v[0:1], v[2:3], v0, v[4:5]
// CHECK-NEXT:{{^}}                                  ^

v_mqsad_pk_u16_u8 v[0:1], v[2:3], v1, v[4:5]
// CHECK: error: destination must be different than all sources
// CHECK-NEXT:{{^}}v_mqsad_pk_u16_u8 v[0:1], v[2:3], v1, v[4:5]
// CHECK-NEXT:{{^}}                                  ^

v_mqsad_pk_u16_u8 v[0:1], v[2:3], v9, v[0:1]
// CHECK: error: destination must be different than all sources
// CHECK-NEXT:{{^}}v_mqsad_pk_u16_u8 v[0:1], v[2:3], v9, v[0:1]
// CHECK-NEXT:{{^}}                                      ^

//==============================================================================
// dim modifier is required on this GPU

image_atomic_add v252, v2, s[8:15]
// CHECK: error: dim modifier is required on this GPU
// CHECK-NEXT:{{^}}image_atomic_add v252, v2, s[8:15]
// CHECK-NEXT:{{^}}^

//==============================================================================
// duplicate data format

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32,BUF_DATA_FORMAT_8]
// CHECK: error: duplicate data format
// CHECK-NEXT:{{^}}tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32,BUF_DATA_FORMAT_8]
// CHECK-NEXT:{{^}}                                                                                ^

//==============================================================================
// duplicate numeric format

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_UINT,BUF_NUM_FORMAT_FLOAT]
// CHECK: error: duplicate numeric format
// CHECK-NEXT:{{^}}tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_UINT,BUF_NUM_FORMAT_FLOAT]
// CHECK-NEXT:{{^}}                                                                                 ^

//==============================================================================
// expected ')' in parentheses expression

v_bfe_u32 v0, 1+(100, v1, v2
// CHECK: error: expected ')' in parentheses expression
// CHECK-NEXT:{{^}}v_bfe_u32 v0, 1+(100, v1, v2
// CHECK-NEXT:{{^}}                    ^

//==============================================================================
// expected a 12-bit signed offset

global_load_dword v1, v[3:4] off, offset:-4097
// CHECK: error: expected a 12-bit signed offset
// CHECK-NEXT:{{^}}global_load_dword v1, v[3:4] off, offset:-4097
// CHECK-NEXT:{{^}}                                  ^

scratch_load_dword v0, v1, off offset:-2049 glc slc
// CHECK: error: expected a 12-bit signed offset
// CHECK-NEXT:{{^}}scratch_load_dword v0, v1, off offset:-2049 glc slc
// CHECK-NEXT:{{^}}                               ^

//==============================================================================
// expected a 16-bit signed jump offset

s_branch 0x10000
// CHECK: error: expected a 16-bit signed jump offset
// CHECK-NEXT:{{^}}s_branch 0x10000
// CHECK-NEXT:{{^}}         ^

//==============================================================================
// expected a 2-bit lane id

ds_swizzle_b32 v8, v2 offset:swizzle(QUAD_PERM, 4, 1, 2, 3)
// CHECK: error: expected a 2-bit lane id
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset:swizzle(QUAD_PERM, 4, 1, 2, 3)
// CHECK-NEXT:{{^}}                                                ^

//==============================================================================
// expected a 20-bit unsigned offset

s_atc_probe_buffer 0x1, s[8:11], -1
// CHECK: error: expected a 20-bit unsigned offset
// CHECK-NEXT:{{^}}s_atc_probe_buffer 0x1, s[8:11], -1
// CHECK-NEXT:{{^}}                                 ^

s_atc_probe_buffer 0x1, s[8:11], 0xFFFFFFFFFFF00000
// CHECK: error: expected a 20-bit unsigned offset
// CHECK-NEXT:{{^}}s_atc_probe_buffer 0x1, s[8:11], 0xFFFFFFFFFFF00000
// CHECK-NEXT:{{^}}                                 ^

s_buffer_atomic_swap s5, s[4:7], 0x1FFFFF
// CHECK: error: expected a 20-bit unsigned offset
// CHECK-NEXT:{{^}}s_buffer_atomic_swap s5, s[4:7], 0x1FFFFF
// CHECK-NEXT:{{^}}                                 ^

//==============================================================================
// expected a 21-bit signed offset

s_atc_probe 0x7, s[4:5], 0x1FFFFF
// CHECK: error: expected a 21-bit signed offset
// CHECK-NEXT:{{^}}s_atc_probe 0x7, s[4:5], 0x1FFFFF
// CHECK-NEXT:{{^}}                         ^

s_atomic_swap s5, s[2:3], 0x1FFFFF
// CHECK: error: expected a 21-bit signed offset
// CHECK-NEXT:{{^}}s_atomic_swap s5, s[2:3], 0x1FFFFF
// CHECK-NEXT:{{^}}                          ^

//==============================================================================
// expected a 2-bit value

v_mov_b32_dpp v5, v1 quad_perm:[3,2,1,4] row_mask:0x0 bank_mask:0x0
// CHECK: error: expected a 2-bit value
// CHECK-NEXT:{{^}}v_mov_b32_dpp v5, v1 quad_perm:[3,2,1,4] row_mask:0x0 bank_mask:0x0
// CHECK-NEXT:{{^}}                                      ^

v_mov_b32_dpp v5, v1 quad_perm:[3,-1,1,3] row_mask:0x0 bank_mask:0x0
// CHECK: error: expected a 2-bit value
// CHECK-NEXT:{{^}}v_mov_b32_dpp v5, v1 quad_perm:[3,-1,1,3] row_mask:0x0 bank_mask:0x0
// CHECK-NEXT:{{^}}                                  ^

//==============================================================================
// expected a 3-bit value

v_mov_b32_dpp v5, v1 dpp8:[-1,1,2,3,4,5,6,7]
// CHECK: error: expected a 3-bit value
// CHECK-NEXT:{{^}}v_mov_b32_dpp v5, v1 dpp8:[-1,1,2,3,4,5,6,7]
// CHECK-NEXT:{{^}}                           ^

v_mov_b32_dpp v5, v1 dpp8:[0,1,2,8,4,5,6,7]
// CHECK: error: expected a 3-bit value
// CHECK-NEXT:{{^}}v_mov_b32_dpp v5, v1 dpp8:[0,1,2,8,4,5,6,7]
// CHECK-NEXT:{{^}}                                 ^

//==============================================================================
// expected a 5-character mask

ds_swizzle_b32 v8, v2 offset:swizzle(BITMASK_PERM, "ppii")
// CHECK: error: expected a 5-character mask
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset:swizzle(BITMASK_PERM, "ppii")
// CHECK-NEXT:{{^}}                                                   ^

//==============================================================================
// expected a closing parentheses

ds_swizzle_b32 v8, v2 offset:swizzle(QUAD_PERM, 0, 1, 2, 3
// CHECK: error: expected a closing parentheses
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset:swizzle(QUAD_PERM, 0, 1, 2, 3
// CHECK-NEXT:{{^}}                                                          ^

ds_swizzle_b32 v8, v2 offset:swizzle(QUAD_PERM, 0, 1, 2, 3, 4)
// CHECK: error: expected a closing parentheses
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset:swizzle(QUAD_PERM, 0, 1, 2, 3, 4)
// CHECK-NEXT:{{^}}                                                          ^

//==============================================================================
// expected a closing parenthesis

s_sendmsg sendmsg(2, 2, 0, 0)
// CHECK: error: expected a closing parenthesis
// CHECK-NEXT:{{^}}s_sendmsg sendmsg(2, 2, 0, 0)
// CHECK-NEXT:{{^}}                         ^

s_waitcnt vmcnt(0
// CHECK: error: expected a closing parenthesis
// CHECK-NEXT:{{^}}s_waitcnt vmcnt(0
// CHECK-NEXT:{{^}}                 ^

//==============================================================================
// expected a closing square bracket

s_mov_b32 s1, s[0 1
// CHECK: error: expected a closing square bracket
// CHECK-NEXT:{{^}}s_mov_b32 s1, s[0 1
// CHECK-NEXT:{{^}}                  ^

s_mov_b32 s1, s[0 s0
// CHECK: error: expected a closing square bracket
// CHECK-NEXT:{{^}}s_mov_b32 s1, s[0 s0
// CHECK-NEXT:{{^}}                  ^

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32,BUF_NUM_FORMAT_FLOAT,BUF_DATA_FORMAT_8]
// CHECK: error: expected a closing square bracket
// CHECK-NEXT:{{^}}tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32,BUF_NUM_FORMAT_FLOAT,BUF_DATA_FORMAT_8]
// CHECK-NEXT:{{^}}                                                                                                    ^

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_UINT
// CHECK: error: expected a closing square bracket
// CHECK-NEXT:{{^}}tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_UINT
// CHECK-NEXT:{{^}}                                                                                ^

v_max3_f16 v5, v1, v2, v3 op_sel:[1,1,1,1 1
// CHECK: error: expected a closing square bracket
// CHECK-NEXT:{{^}}v_max3_f16 v5, v1, v2, v3 op_sel:[1,1,1,1 1
// CHECK-NEXT:{{^}}                                          ^

v_max3_f16 v5, v1, v2, v3 op_sel:[1,1,1,1,
// CHECK: error: expected a closing square bracket
// CHECK-NEXT:{{^}}v_max3_f16 v5, v1, v2, v3 op_sel:[1,1,1,1,
// CHECK-NEXT:{{^}}                                         ^

v_max3_f16 v5, v1, v2, v3 op_sel:[1,1,1,1[
// CHECK: error: expected a closing square bracket
// CHECK-NEXT:{{^}}v_max3_f16 v5, v1, v2, v3 op_sel:[1,1,1,1[
// CHECK-NEXT:{{^}}                                         ^

v_pk_add_u16 v1, v2, v3 op_sel:[0,0,0,0,0]
// CHECK: error: expected a closing square bracket
// CHECK-NEXT:{{^}}v_pk_add_u16 v1, v2, v3 op_sel:[0,0,0,0,0]
// CHECK-NEXT:{{^}}                                       ^

v_mov_b32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7)
// CHECK: error: expected a closing square bracket
// CHECK-NEXT:{{^}}v_mov_b32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7)
// CHECK-NEXT:{{^}}                                          ^

v_mov_b32_dpp v5, v1 quad_perm:[3,2,1,0) row_mask:0x0 bank_mask:0x0
// CHECK: error: expected a closing square bracket
// CHECK-NEXT:{{^}}v_mov_b32_dpp v5, v1 quad_perm:[3,2,1,0) row_mask:0x0 bank_mask:0x0
// CHECK-NEXT:{{^}}                                       ^

//==============================================================================
// expected a colon

ds_swizzle_b32 v8, v2 offset
// CHECK: error: expected a colon
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset
// CHECK-NEXT:{{^}}                            ^

ds_swizzle_b32 v8, v2 offset-
// CHECK: error: expected a colon
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset-
// CHECK-NEXT:{{^}}                            ^

//==============================================================================
// expected a comma

ds_swizzle_b32 v8, v2 offset:swizzle(QUAD_PERM
// CHECK: error: expected a comma
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset:swizzle(QUAD_PERM
// CHECK-NEXT:{{^}}                                              ^

ds_swizzle_b32 v8, v2 offset:swizzle(QUAD_PERM, 0, 1, 2)
// CHECK: error: expected a comma
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset:swizzle(QUAD_PERM, 0, 1, 2)
// CHECK-NEXT:{{^}}                                                       ^

s_setreg_b32  hwreg(1,2 3), s2
// CHECK: error: expected a comma
// CHECK-NEXT:{{^}}s_setreg_b32  hwreg(1,2 3), s2
// CHECK-NEXT:{{^}}                        ^

v_pk_add_u16 v1, v2, v3 op_sel:[0 0]
// CHECK: error: expected a comma
// CHECK-NEXT:{{^}}v_pk_add_u16 v1, v2, v3 op_sel:[0 0]
// CHECK-NEXT:{{^}}                                  ^

v_mov_b32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6]
// CHECK: error: expected a comma
// CHECK-NEXT:{{^}}v_mov_b32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6]
// CHECK-NEXT:{{^}}                                        ^

v_mov_b32_dpp v5, v1 quad_perm:[3,2] row_mask:0x0 bank_mask:0x0
// CHECK: error: expected a comma
// CHECK-NEXT:{{^}}v_mov_b32_dpp v5, v1 quad_perm:[3,2] row_mask:0x0 bank_mask:0x0
// CHECK-NEXT:{{^}}                                   ^

//==============================================================================
// expected a comma or a closing parenthesis

s_setreg_b32  hwreg(1 2,3), s2
// CHECK: error: expected a comma or a closing parenthesis
// CHECK-NEXT:{{^}}s_setreg_b32  hwreg(1 2,3), s2
// CHECK-NEXT:{{^}}                      ^

//==============================================================================
// expected a comma or a closing square bracket

s_mov_b64 s[10:11], [s0
// CHECK: error: expected a comma or a closing square bracket
// CHECK-NEXT:{{^}}s_mov_b64 s[10:11], [s0
// CHECK-NEXT:{{^}}                       ^

s_mov_b64 s[10:11], [s0,s1
// CHECK: error: expected a comma or a closing square bracket
// CHECK-NEXT:{{^}}s_mov_b64 s[10:11], [s0,s1
// CHECK-NEXT:{{^}}                          ^

image_load_mip v[253:255], [v255, v254 dmask:0xe dim:1D
// CHECK: error: expected a comma or a closing square bracket
// CHECK-NEXT:{{^}}image_load_mip v[253:255], [v255, v254 dmask:0xe dim:1D
// CHECK-NEXT:{{^}}                                       ^

image_load_mip v[253:255], [v255, v254
// CHECK: error: expected a comma or a closing square bracket
// CHECK-NEXT:{{^}}image_load_mip v[253:255], [v255, v254
// CHECK-NEXT:{{^}}                                      ^

//==============================================================================
// expected a counter name

s_waitcnt vmcnt(0) & expcnt(0) & 1
// CHECK: error: expected a counter name
// CHECK-NEXT:{{^}}s_waitcnt vmcnt(0) & expcnt(0) & 1
// CHECK-NEXT:{{^}}                                 ^

s_waitcnt vmcnt(0) & expcnt(0) & lgkmcnt(0)&
// CHECK: error: expected a counter name
// CHECK-NEXT:{{^}}s_waitcnt vmcnt(0) & expcnt(0) & lgkmcnt(0)&
// CHECK-NEXT:{{^}}                                            ^

s_waitcnt vmcnt(0) & expcnt(0) 1
// CHECK: error: expected a counter name
// CHECK-NEXT:{{^}}s_waitcnt vmcnt(0) & expcnt(0) 1
// CHECK-NEXT:{{^}}                               ^

s_waitcnt vmcnt(0), expcnt(0), lgkmcnt(0),
// CHECK: error: expected a counter name
// CHECK-NEXT:{{^}}s_waitcnt vmcnt(0), expcnt(0), lgkmcnt(0),
// CHECK-NEXT:{{^}}                                          ^

//==============================================================================
// expected a format string

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[]
// CHECK: error: expected a format string
// CHECK-NEXT:{{^}}tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[]
// CHECK-NEXT:{{^}}                                                             ^

//==============================================================================
// expected a left parenthesis

s_waitcnt vmcnt(0) & expcnt(0) & x
// CHECK: error: expected a left parenthesis
// CHECK-NEXT:{{^}}s_waitcnt vmcnt(0) & expcnt(0) & x
// CHECK-NEXT:{{^}}                                  ^

//==============================================================================
// expected a left square bracket

v_pk_add_u16 v1, v2, v3 op_sel:
// CHECK: error: expected a left square bracket
// CHECK-NEXT:{{^}}v_pk_add_u16 v1, v2, v3 op_sel:
// CHECK-NEXT:{{^}}                               ^

//==============================================================================
// expected a register

image_load v[0:3], [v4, v5, 6], s[8:15] dmask:0xf dim:3D unorm
// CHECK: error: expected a register
// CHECK-NEXT:{{^}}image_load v[0:3], [v4, v5, 6], s[8:15] dmask:0xf dim:3D unorm
// CHECK-NEXT:{{^}}                            ^

image_load v[0:3], [v4, v5, v], s[8:15] dmask:0xf dim:3D unorm
// CHECK: error: expected a register
// CHECK-NEXT:{{^}}image_load v[0:3], [v4, v5, v], s[8:15] dmask:0xf dim:3D unorm
// CHECK-NEXT:{{^}}                            ^

//==============================================================================
// expected a register or a list of registers

s_mov_b32 s1, [s0, 1
// CHECK: error: expected a register or a list of registers
// CHECK-NEXT:{{^}}s_mov_b32 s1, [s0, 1
// CHECK-NEXT:{{^}}                   ^

//==============================================================================
// expected a single 32-bit register

s_mov_b64 s[10:11], [s0,s[2:3]]
// CHECK: error: expected a single 32-bit register
// CHECK-NEXT:{{^}}s_mov_b64 s[10:11], [s0,s[2:3]]
// CHECK-NEXT:{{^}}                        ^

//==============================================================================
// expected a string

ds_swizzle_b32 v8, v2 offset:swizzle(BITMASK_PERM, pppii)
// CHECK: error: expected a string
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset:swizzle(BITMASK_PERM, pppii)
// CHECK-NEXT:{{^}}                                                   ^

//==============================================================================
// expected a swizzle mode

ds_swizzle_b32 v8, v2 offset:swizzle(XXX,1)
// CHECK: error: expected a swizzle mode
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset:swizzle(XXX,1)
// CHECK-NEXT:{{^}}                                     ^

//==============================================================================
// expected absolute expression

s_waitcnt vmcnt(x)
// CHECK: error: expected absolute expression
// CHECK-NEXT:{{^}}s_waitcnt vmcnt(x)
// CHECK-NEXT:{{^}}                ^

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], format:[BUF_DATA_FORMAT_32]
// CHECK: error: expected absolute expression
// CHECK-NEXT:{{^}}tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], format:[BUF_DATA_FORMAT_32]
// CHECK-NEXT:{{^}}                                                         ^

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format: offset:52
// CHECK: error: expected absolute expression
// CHECK-NEXT:{{^}}tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format: offset:52
// CHECK-NEXT:{{^}}                                                             ^

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:BUF_NUM_FORMAT_UINT]
// CHECK: error: expected absolute expression
// CHECK-NEXT:{{^}}tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:BUF_NUM_FORMAT_UINT]
// CHECK-NEXT:{{^}}                                                            ^

v_mov_b32_dpp v5, v1 dpp8:[0,1,2,x,4,5,6,7]
// CHECK: error: expected absolute expression
// CHECK-NEXT:{{^}}v_mov_b32_dpp v5, v1 dpp8:[0,1,2,x,4,5,6,7]
// CHECK-NEXT:{{^}}                                 ^

v_mov_b32_dpp v5, v1 quad_perm:[3,x,1,0] row_mask:0x0 bank_mask:0x0
// CHECK: error: expected absolute expression
// CHECK-NEXT:{{^}}v_mov_b32_dpp v5, v1 quad_perm:[3,x,1,0] row_mask:0x0 bank_mask:0x0
// CHECK-NEXT:{{^}}                                  ^

v_mov_b32_dpp v5, v1 row_share:x row_mask:0x0 bank_mask:0x0
// CHECK: error: expected absolute expression
// CHECK-NEXT:{{^}}v_mov_b32_dpp v5, v1 row_share:x row_mask:0x0 bank_mask:0x0
// CHECK-NEXT:{{^}}                               ^

//==============================================================================
// expected a message name or an absolute expression

s_sendmsg sendmsg(MSG_GSX, GS_OP_CUT, 0)
// CHECK: error: expected a message name or an absolute expression
// CHECK-NEXT:{{^}}s_sendmsg sendmsg(MSG_GSX, GS_OP_CUT, 0)
// CHECK-NEXT:{{^}}                  ^

//==============================================================================
// expected a register name or an absolute expression

s_setreg_b32  hwreg(HW_REG_WRONG), s2
// CHECK: error: expected a register name or an absolute expression
// CHECK-NEXT:{{^}}s_setreg_b32  hwreg(HW_REG_WRONG), s2
// CHECK-NEXT:{{^}}                    ^

//==============================================================================
// expected a sendmsg macro or an absolute expression

s_sendmsg undef
// CHECK: error: expected a sendmsg macro or an absolute expression
// CHECK-NEXT:{{^}}s_sendmsg undef
// CHECK-NEXT:{{^}}          ^

//==============================================================================
// expected a swizzle macro or an absolute expression

ds_swizzle_b32 v8, v2 offset:SWZ(QUAD_PERM, 0, 1, 2, 3)
// CHECK: error: expected a swizzle macro or an absolute expression
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset:SWZ(QUAD_PERM, 0, 1, 2, 3)
// CHECK-NEXT:{{^}}                             ^

//==============================================================================
// expected a hwreg macro or an absolute expression

s_setreg_b32 undef, s2
// CHECK: error: expected a hwreg macro or an absolute expression
// CHECK-NEXT:{{^}}s_setreg_b32 undef, s2
// CHECK-NEXT:{{^}}             ^

//==============================================================================
// expected an 11-bit unsigned offset

flat_atomic_cmpswap v0, v[1:2], v[3:4] offset:4095 glc
// CHECK: error: expected a 11-bit unsigned offset
// CHECK-NEXT:{{^}}flat_atomic_cmpswap v0, v[1:2], v[3:4] offset:4095 glc
// CHECK-NEXT:{{^}}                                       ^

//==============================================================================
// expected an absolute expression

v_ceil_f32 v1, abs(u)
// CHECK: error: expected an absolute expression
// CHECK-NEXT:{{^}}v_ceil_f32 v1, abs(u)
// CHECK-NEXT:{{^}}                   ^

v_ceil_f32 v1, neg(u)
// CHECK: error: expected an absolute expression
// CHECK-NEXT:{{^}}v_ceil_f32 v1, neg(u)
// CHECK-NEXT:{{^}}                   ^

v_ceil_f32 v1, |u|
// CHECK: error: expected an absolute expression
// CHECK-NEXT:{{^}}v_ceil_f32 v1, |u|
// CHECK-NEXT:{{^}}                ^

v_mov_b32_sdwa v1, sext(u)
// CHECK: error: expected an absolute expression
// CHECK-NEXT:{{^}}v_mov_b32_sdwa v1, sext(u)
// CHECK-NEXT:{{^}}                        ^

//==============================================================================
// expected an identifier

v_mov_b32_sdwa v5, v1 dst_sel:
// CHECK: error: expected an identifier
// CHECK-NEXT:{{^}}v_mov_b32_sdwa v5, v1 dst_sel:
// CHECK-NEXT:{{^}}                              ^

v_mov_b32_sdwa v5, v1 dst_sel:0
// CHECK: error: expected an identifier
// CHECK-NEXT:{{^}}v_mov_b32_sdwa v5, v1 dst_sel:0
// CHECK-NEXT:{{^}}                              ^

v_mov_b32_sdwa v5, v1 dst_sel:DWORD dst_unused:[UNUSED_PAD]
// CHECK: error: expected an identifier
// CHECK-NEXT:{{^}}v_mov_b32_sdwa v5, v1 dst_sel:DWORD dst_unused:[UNUSED_PAD]
// CHECK-NEXT:{{^}}                                               ^

//==============================================================================
// expected an opening square bracket

v_mov_b32_dpp v5, v1 dpp8:(0,1,2,3,4,5,6,7)
// CHECK: error: expected an opening square bracket
// CHECK-NEXT:{{^}}v_mov_b32_dpp v5, v1 dpp8:(0,1,2,3,4,5,6,7)
// CHECK-NEXT:{{^}}                          ^

v_mov_b32_dpp v5, v1 quad_perm:(3,2,1,0) row_mask:0x0 bank_mask:0x0
// CHECK: error: expected an opening square bracket
// CHECK-NEXT:{{^}}v_mov_b32_dpp v5, v1 quad_perm:(3,2,1,0) row_mask:0x0 bank_mask:0x0
// CHECK-NEXT:{{^}}                               ^

//==============================================================================
// expected an operation name or an absolute expression

s_sendmsg sendmsg(MSG_GS, GS_OP_CUTX, 0)
// CHECK: error: expected an operation name or an absolute expression
// CHECK-NEXT:{{^}}s_sendmsg sendmsg(MSG_GS, GS_OP_CUTX, 0)
// CHECK-NEXT:{{^}}                          ^

//==============================================================================
// failed parsing operand.

v_ceil_f16 v0, abs(neg(1))
// CHECK: error: failed parsing operand.
// CHECK-NEXT:{{^}}v_ceil_f16 v0, abs(neg(1))
// CHECK-NEXT:{{^}}                   ^

//==============================================================================
// first register index should not exceed second index

s_mov_b64 s[10:11], s[1:0]
// CHECK: error: first register index should not exceed second index
// CHECK-NEXT:{{^}}s_mov_b64 s[10:11], s[1:0]
// CHECK-NEXT:{{^}}                      ^

//==============================================================================
// group size must be a power of two

ds_swizzle_b32 v8, v2 offset:swizzle(BROADCAST,3,1)
// CHECK: error: group size must be a power of two
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset:swizzle(BROADCAST,3,1)
// CHECK-NEXT:{{^}}                                               ^

ds_swizzle_b32 v8, v2 offset:swizzle(REVERSE,3)
// CHECK: error: group size must be a power of two
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset:swizzle(REVERSE,3)
// CHECK-NEXT:{{^}}                                             ^

ds_swizzle_b32 v8, v2 offset:swizzle(SWAP,3)
// CHECK: error: group size must be a power of two
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset:swizzle(SWAP,3)
// CHECK-NEXT:{{^}}                                          ^

//==============================================================================
// group size must be in the interval [1,16]

ds_swizzle_b32 v8, v2 offset:swizzle(SWAP,0)
// CHECK: error: group size must be in the interval [1,16]
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset:swizzle(SWAP,0)
// CHECK-NEXT:{{^}}                                          ^

//==============================================================================
// group size must be in the interval [2,32]

ds_swizzle_b32 v8, v2 offset:swizzle(BROADCAST,1,0)
// CHECK: error: group size must be in the interval [2,32]
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset:swizzle(BROADCAST,1,0)
// CHECK-NEXT:{{^}}                                               ^

//==============================================================================
// image address size does not match dim and a16

image_load v[0:3], v[0:1], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_3D
// CHECK: error: image address size does not match dim and a16
// CHECK-NEXT:{{^}}image_load v[0:3], v[0:1], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_3D
// CHECK-NEXT:{{^}}^

//==============================================================================
// image data size does not match dmask and tfe

image_load v[0:1], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D
// CHECK: error: image data size does not match dmask and tfe
// CHECK-NEXT:{{^}}image_load v[0:1], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D
// CHECK-NEXT:{{^}}^

//==============================================================================
// instruction must use glc

flat_atomic_cmpswap v0, v[1:2], v[3:4] offset:2047
// CHECK: error: instruction must use glc
// CHECK-NEXT:{{^}}flat_atomic_cmpswap v0, v[1:2], v[3:4] offset:2047
// CHECK-NEXT:{{^}}^

//==============================================================================
// instruction not supported on this GPU

s_cbranch_join 1
// CHECK: error: instruction not supported on this GPU
// CHECK-NEXT:{{^}}s_cbranch_join 1
// CHECK-NEXT:{{^}}^

//==============================================================================
// invalid bit offset: only 5-bit values are legal

s_getreg_b32  s2, hwreg(3,32,32)
// CHECK: error: invalid bit offset: only 5-bit values are legal
// CHECK-NEXT:{{^}}s_getreg_b32  s2, hwreg(3,32,32)
// CHECK-NEXT:{{^}}                          ^

//==============================================================================
// invalid bitfield width: only values from 1 to 32 are legal

s_setreg_b32  hwreg(3,0,33), s2
// CHECK: error: invalid bitfield width: only values from 1 to 32 are legal
// CHECK-NEXT:{{^}}s_setreg_b32  hwreg(3,0,33), s2
// CHECK-NEXT:{{^}}                        ^

//==============================================================================
// invalid code of hardware register: only 6-bit values are legal

s_setreg_b32  hwreg(0x40), s2
// CHECK: error: invalid code of hardware register: only 6-bit values are legal
// CHECK-NEXT:{{^}}s_setreg_b32  hwreg(0x40), s2
// CHECK-NEXT:{{^}}                    ^

//==============================================================================
// invalid counter name x

s_waitcnt vmcnt(0) & expcnt(0) x(0)
// CHECK: error: invalid counter name x
// CHECK-NEXT:{{^}}s_waitcnt vmcnt(0) & expcnt(0) x(0)
// CHECK-NEXT:{{^}}                               ^

//==============================================================================
// invalid dim value

image_load v[0:1], v0, s[0:7] dmask:0x9 dim:1 D
// CHECK: error: invalid dim value
// CHECK-NEXT:{{^}}image_load v[0:1], v0, s[0:7] dmask:0x9 dim:1 D
// CHECK-NEXT:{{^}}                                            ^

image_atomic_xor v4, v32, s[96:103] dmask:0x1 dim:, glc
// CHECK: error: invalid dim value
// CHECK-NEXT:{{^}}image_atomic_xor v4, v32, s[96:103] dmask:0x1 dim:, glc
// CHECK-NEXT:{{^}}                                                  ^

image_load v[0:1], v0, s[0:7] dmask:0x9 dim:7D
// CHECK: error: invalid dim value
// CHECK-NEXT:{{^}}image_load v[0:1], v0, s[0:7] dmask:0x9 dim:7D
// CHECK-NEXT:{{^}}                                            ^

//==============================================================================
// invalid dst_sel value

v_mov_b32_sdwa v5, v1 dst_sel:WORD
// CHECK: error: invalid dst_sel value
// CHECK-NEXT:{{^}}v_mov_b32_sdwa v5, v1 dst_sel:WORD
// CHECK-NEXT:{{^}}                              ^

//==============================================================================
// invalid dst_unused value

v_mov_b32_sdwa v5, v1 dst_unused:UNUSED
// CHECK: error: invalid dst_unused value
// CHECK-NEXT:{{^}}v_mov_b32_sdwa v5, v1 dst_unused:UNUSED
// CHECK-NEXT:{{^}}                                 ^

//==============================================================================
// invalid exp target

exp invalid_target_10 v3, v2, v1, v0
// CHECK: error: invalid exp target
// CHECK-NEXT:{{^}}exp invalid_target_10 v3, v2, v1, v0
// CHECK-NEXT:{{^}}    ^

exp pos00 v3, v2, v1, v0
// CHECK: error: invalid exp target
// CHECK-NEXT:{{^}}exp pos00 v3, v2, v1, v0
// CHECK-NEXT:{{^}}    ^

//==============================================================================
// invalid immediate: only 16-bit values are legal

s_setreg_b32  0x1f803, s2
// CHECK: error: invalid immediate: only 16-bit values are legal
// CHECK-NEXT:{{^}}s_setreg_b32  0x1f803, s2
// CHECK-NEXT:{{^}}              ^

//==============================================================================
// invalid instruction

v_dot_f32_f16 v0, v1, v2
// CHECK: error: invalid instruction
// CHECK-NEXT:{{^}}v_dot_f32_f16 v0, v1, v2
// CHECK-NEXT:{{^}}^

//==============================================================================
// invalid interpolation attribute

v_interp_p2_f32 v0, v1, att
// CHECK: error: invalid interpolation attribute
// CHECK-NEXT:{{^}}v_interp_p2_f32 v0, v1, att
// CHECK-NEXT:{{^}}                        ^

//==============================================================================
// invalid interpolation slot

v_interp_mov_f32 v8, p1, attr0.x
// CHECK: error: invalid interpolation slot
// CHECK-NEXT:{{^}}v_interp_mov_f32 v8, p1, attr0.x
// CHECK-NEXT:{{^}}                     ^

//==============================================================================
// invalid mask

ds_swizzle_b32 v8, v2 offset:swizzle(BITMASK_PERM, "pppi2")
// CHECK: error: invalid mask
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset:swizzle(BITMASK_PERM, "pppi2")
// CHECK-NEXT:{{^}}                                                   ^

//==============================================================================
// invalid message id

s_sendmsg sendmsg(-1)
// CHECK: error: invalid message id
// CHECK-NEXT:{{^}}s_sendmsg sendmsg(-1)
// CHECK-NEXT:{{^}}                  ^

//==============================================================================
// invalid message stream id

s_sendmsg sendmsg(2, 2, 4)
// CHECK: error: invalid message stream id
// CHECK-NEXT:{{^}}s_sendmsg sendmsg(2, 2, 4)
// CHECK-NEXT:{{^}}                        ^

s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 4)
// CHECK: error: invalid message stream id
// CHECK-NEXT:{{^}}s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 4)
// CHECK-NEXT:{{^}}                                     ^

//==============================================================================
// invalid mul value.

v_cvt_f64_i32 v[5:6], s1 mul:3
// CHECK: error: invalid mul value.
// CHECK-NEXT:{{^}}v_cvt_f64_i32 v[5:6], s1 mul:3
// CHECK-NEXT:{{^}}                         ^

//==============================================================================
// invalid or missing interpolation attribute channel

v_interp_p2_f32 v0, v1, attr0.q
// CHECK: error: invalid or missing interpolation attribute channel
// CHECK-NEXT:{{^}}v_interp_p2_f32 v0, v1, attr0.q
// CHECK-NEXT:{{^}}                        ^

//==============================================================================
// invalid or missing interpolation attribute number

v_interp_p2_f32 v7, v1, attr.x
// CHECK: error: invalid or missing interpolation attribute number
// CHECK-NEXT:{{^}}v_interp_p2_f32 v7, v1, attr.x
// CHECK-NEXT:{{^}}                        ^

//==============================================================================
// invalid op_sel operand

v_permlane16_b32 v5, v1, s2, s3 op_sel:[0, 0, 0, 1]
// CHECK: error: invalid op_sel operand
// CHECK-NEXT:{{^}}v_permlane16_b32 v5, v1, s2, s3 op_sel:[0, 0, 0, 1]
// CHECK-NEXT:{{^}}                                ^

//==============================================================================
// invalid op_sel value.

v_pk_add_u16 v1, v2, v3 op_sel:[-1,0]
// CHECK: error: invalid op_sel value.
// CHECK-NEXT:{{^}}v_pk_add_u16 v1, v2, v3 op_sel:[-1,0]
// CHECK-NEXT:{{^}}                                ^

//==============================================================================
// invalid operand (violates constant bus restrictions)

v_ashrrev_i64 v[0:1], 0x100, s[0:1]
// CHECK: error: invalid operand (violates constant bus restrictions)
// CHECK-NEXT:{{^}}v_ashrrev_i64 v[0:1], 0x100, s[0:1]
// CHECK-NEXT:{{^}}                             ^

v_ashrrev_i64 v[0:1], s3, s[0:1]
// CHECK: error: invalid operand (violates constant bus restrictions)
// CHECK-NEXT:{{^}}v_ashrrev_i64 v[0:1], s3, s[0:1]
// CHECK-NEXT:{{^}}                          ^

v_bfe_u32 v0, s1, 0x3039, s2
// CHECK: error: invalid operand (violates constant bus restrictions)
// CHECK-NEXT:{{^}}v_bfe_u32 v0, s1, 0x3039, s2
// CHECK-NEXT:{{^}}                          ^

v_bfe_u32 v0, s1, s2, s3
// CHECK: error: invalid operand (violates constant bus restrictions)
// CHECK-NEXT:{{^}}v_bfe_u32 v0, s1, s2, s3
// CHECK-NEXT:{{^}}                      ^

v_div_fmas_f32 v5, s3, 0x123, v3
// CHECK: error: invalid operand (violates constant bus restrictions)
// CHECK-NEXT:{{^}}v_div_fmas_f32 v5, s3, 0x123, v3
// CHECK-NEXT:{{^}}                       ^

v_div_fmas_f32 v5, s3, v3, 0x123
// CHECK: error: invalid operand (violates constant bus restrictions)
// CHECK-NEXT:{{^}}v_div_fmas_f32 v5, s3, v3, 0x123
// CHECK-NEXT:{{^}}                           ^

v_div_fmas_f32 v5, 0x123, v3, s3
// CHECK: error: invalid operand (violates constant bus restrictions)
// CHECK-NEXT:{{^}}v_div_fmas_f32 v5, 0x123, v3, s3
// CHECK-NEXT:{{^}}                              ^

v_div_fmas_f32 v5, s3, s4, v3
// CHECK: error: invalid operand (violates constant bus restrictions)
// CHECK-NEXT:{{^}}v_div_fmas_f32 v5, s3, s4, v3
// CHECK-NEXT:{{^}}                       ^

//==============================================================================
// invalid operand for instruction

buffer_load_dword v5, off, s[8:11], s3 tfe lds
// CHECK: error: invalid operand for instruction
// CHECK-NEXT:{{^}}buffer_load_dword v5, off, s[8:11], s3 tfe lds
// CHECK-NEXT:{{^}}                                           ^

exp mrt0 0x12345678, v0, v0, v0
// CHECK: error: invalid operand for instruction
// CHECK-NEXT:{{^}}exp mrt0 0x12345678, v0, v0, v0
// CHECK-NEXT:{{^}}         ^

v_cmp_eq_f32 s[0:1], private_base, s0
// CHECK: error: invalid operand for instruction
// CHECK-NEXT:{{^}}v_cmp_eq_f32 s[0:1], private_base, s0
// CHECK-NEXT:{{^}}             ^

//==============================================================================
// invalid operation id

s_sendmsg sendmsg(15, -1)
// CHECK: error: invalid operation id
// CHECK-NEXT:{{^}}s_sendmsg sendmsg(15, -1)
// CHECK-NEXT:{{^}}                      ^

//==============================================================================
// invalid or unsupported register size

s_mov_b64 s[0:17], -1
// CHECK: error: invalid or unsupported register size
// CHECK-NEXT:{{^}}s_mov_b64 s[0:17], -1
// CHECK-NEXT:{{^}}          ^

//==============================================================================
// invalid register alignment

s_load_dwordx4 s[1:4], s[2:3], s4
// CHECK: error: invalid register alignment
// CHECK-NEXT:{{^}}s_load_dwordx4 s[1:4], s[2:3], s4
// CHECK-NEXT:{{^}}               ^

//==============================================================================
// invalid register index

s_mov_b32 s1, s[0:-1]
// CHECK: error: invalid register index
// CHECK-NEXT:{{^}}s_mov_b32 s1, s[0:-1]
// CHECK-NEXT:{{^}}                  ^

v_add_f64 v[0:1], v[0:1], v[0xF00000001:0x2]
// CHECK: error: invalid register index
// CHECK-NEXT:{{^}}v_add_f64 v[0:1], v[0:1], v[0xF00000001:0x2]
// CHECK-NEXT:{{^}}                            ^

//==============================================================================
// invalid register name

s_mov_b64 s[10:11], [x0,s1]
// CHECK: error: invalid register name
// CHECK-NEXT:{{^}}s_mov_b64 s[10:11], [x0,s1]
// CHECK-NEXT:{{^}}                     ^

//==============================================================================
// invalid row_share value

v_mov_b32_dpp v5, v1 row_share:16 row_mask:0x0 bank_mask:0x0
// CHECK: error: invalid row_share value
// CHECK-NEXT:{{^}}v_mov_b32_dpp v5, v1 row_share:16 row_mask:0x0 bank_mask:0x0
// CHECK-NEXT:{{^}}                               ^

v_mov_b32_dpp v5, v1 row_share:-1 row_mask:0x0 bank_mask:0x0
// CHECK: error: invalid row_share value
// CHECK-NEXT:{{^}}v_mov_b32_dpp v5, v1 row_share:-1 row_mask:0x0 bank_mask:0x0
// CHECK-NEXT:{{^}}                               ^

//==============================================================================
// invalid syntax, expected 'neg' modifier

v_ceil_f32 v0, --1
// CHECK: error: invalid syntax, expected 'neg' modifier
// CHECK-NEXT:{{^}}v_ceil_f32 v0, --1
// CHECK-NEXT:{{^}}               ^

//==============================================================================
// lane id must be in the interval [0,group size - 1]

ds_swizzle_b32 v8, v2 offset:swizzle(BROADCAST,2,-1)
// CHECK: error: lane id must be in the interval [0,group size - 1]
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset:swizzle(BROADCAST,2,-1)
// CHECK-NEXT:{{^}}                                                 ^

//==============================================================================
// lds_direct cannot be used with this instruction

v_ashrrev_i16 v0, lds_direct, v0
// CHECK: error: lds_direct cannot be used with this instruction
// CHECK-NEXT:{{^}}v_ashrrev_i16 v0, lds_direct, v0
// CHECK-NEXT:{{^}}                  ^

v_ashrrev_i16 v0, v1, lds_direct
// CHECK: error: lds_direct cannot be used with this instruction
// CHECK-NEXT:{{^}}v_ashrrev_i16 v0, v1, lds_direct
// CHECK-NEXT:{{^}}                      ^

v_mov_b32_sdwa v1, src_lds_direct dst_sel:DWORD
// CHECK: error: lds_direct cannot be used with this instruction
// CHECK-NEXT:{{^}}v_mov_b32_sdwa v1, src_lds_direct dst_sel:DWORD
// CHECK-NEXT:{{^}}                   ^

v_add_f32_sdwa v5, v1, lds_direct dst_sel:DWORD
// CHECK: error: lds_direct cannot be used with this instruction
// CHECK-NEXT:{{^}}v_add_f32_sdwa v5, v1, lds_direct dst_sel:DWORD
// CHECK-NEXT:{{^}}                       ^

//==============================================================================
// lds_direct may be used as src0 only

v_add_f32 v5, v1, lds_direct
// CHECK: error: lds_direct may be used as src0 only
// CHECK-NEXT:{{^}}v_add_f32 v5, v1, lds_direct
// CHECK-NEXT:{{^}}                  ^

//==============================================================================
// message does not support operations

s_sendmsg sendmsg(MSG_GS_ALLOC_REQ, 0)
// CHECK: error: message does not support operations
// CHECK-NEXT:{{^}}s_sendmsg sendmsg(MSG_GS_ALLOC_REQ, 0)
// CHECK-NEXT:{{^}}                                    ^

//==============================================================================
// message operation does not support streams

s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP, 0)
// CHECK: error: message operation does not support streams
// CHECK-NEXT:{{^}}s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP, 0)
// CHECK-NEXT:{{^}}                                          ^

//==============================================================================
// missing message operation

s_sendmsg sendmsg(MSG_SYSMSG)
// CHECK: error: missing message operation
// CHECK-NEXT:{{^}}s_sendmsg sendmsg(MSG_SYSMSG)
// CHECK-NEXT:{{^}}                  ^

//==============================================================================
// missing register index

s_mov_b64 s[10:11], [s
// CHECK: error: missing register index
// CHECK-NEXT:{{^}}s_mov_b64 s[10:11], [s
// CHECK-NEXT:{{^}}                      ^

s_mov_b64 s[10:11], [s,s1]
// CHECK: error: missing register index
// CHECK-NEXT:{{^}}s_mov_b64 s[10:11], [s,s1]
// CHECK-NEXT:{{^}}                      ^

//==============================================================================
// not a valid operand.

s_branch offset:1
// CHECK: error: not a valid operand.
// CHECK-NEXT:{{^}}s_branch offset:1
// CHECK-NEXT:{{^}}         ^

v_mov_b32 v0, v0 row_bcast:0
// CHECK: error: not a valid operand.
// CHECK-NEXT:{{^}}v_mov_b32 v0, v0 row_bcast:0
// CHECK-NEXT:{{^}}                 ^

//==============================================================================
// only one literal operand is allowed

s_and_b32 s2, 0x12345678, 0x12345679
// CHECK: error: only one literal operand is allowed
// CHECK-NEXT:{{^}}s_and_b32 s2, 0x12345678, 0x12345679
// CHECK-NEXT:{{^}}                          ^

v_add_f64 v[0:1], 1.23456, -abs(1.2345)
// CHECK: error: only one literal operand is allowed
// CHECK-NEXT:{{^}}v_add_f64 v[0:1], 1.23456, -abs(1.2345)
// CHECK-NEXT:{{^}}                                ^

v_min3_i16 v5, 0x5678, 0x5678, 0x5679
// CHECK: error: only one literal operand is allowed
// CHECK-NEXT:{{^}}v_min3_i16 v5, 0x5678, 0x5678, 0x5679
// CHECK-NEXT:{{^}}                               ^

v_pk_add_f16 v1, 25.0, 25.1
// CHECK: error: only one literal operand is allowed
// CHECK-NEXT:{{^}}v_pk_add_f16 v1, 25.0, 25.1
// CHECK-NEXT:{{^}}                       ^

v_fma_mix_f32 v5, 0x7c, 0x7b, 1
// CHECK: error: only one literal operand is allowed
// CHECK-NEXT:{{^}}v_fma_mix_f32 v5, 0x7c, 0x7b, 1
// CHECK-NEXT:{{^}}                        ^

v_pk_add_i16 v5, 0x7c, 0x4000
// CHECK: error: only one literal operand is allowed
// CHECK-NEXT:{{^}}v_pk_add_i16 v5, 0x7c, 0x4000
// CHECK-NEXT:{{^}}                       ^

v_pk_add_i16 v5, 0x4400, 0x4000
// CHECK: error: only one literal operand is allowed
// CHECK-NEXT:{{^}}v_pk_add_i16 v5, 0x4400, 0x4000
// CHECK-NEXT:{{^}}                         ^

v_bfe_u32 v0, v2, 123, undef
// CHECK: error: only one literal operand is allowed
// CHECK-NEXT:{{^}}v_bfe_u32 v0, v2, 123, undef
// CHECK-NEXT:{{^}}                       ^

v_bfe_u32 v0, v2, undef, 123
// CHECK: error: only one literal operand is allowed
// CHECK-NEXT:{{^}}v_bfe_u32 v0, v2, undef, 123
// CHECK-NEXT:{{^}}                         ^

//==============================================================================
// out of bounds interpolation attribute number

v_interp_p1_f32 v0, v1, attr64.w
// CHECK: error: out of bounds interpolation attribute number
// CHECK-NEXT:{{^}}v_interp_p1_f32 v0, v1, attr64.w
// CHECK-NEXT:{{^}}                        ^

//==============================================================================
// out of range format

tbuffer_load_format_d16_x v0, off, s[0:3], format:-1, 0
// CHECK: error: out of range format
// CHECK-NEXT:{{^}}tbuffer_load_format_d16_x v0, off, s[0:3], format:-1, 0
// CHECK-NEXT:{{^}}                                           ^

//==============================================================================
// register does not fit in the list

s_mov_b64 s[10:11], [exec,exec_lo]
// CHECK: error: register does not fit in the list
// CHECK-NEXT:{{^}}s_mov_b64 s[10:11], [exec,exec_lo]
// CHECK-NEXT:{{^}}                          ^

s_mov_b64 s[10:11], [exec_lo,exec]
// CHECK: error: register does not fit in the list
// CHECK-NEXT:{{^}}s_mov_b64 s[10:11], [exec_lo,exec]
// CHECK-NEXT:{{^}}                             ^

//==============================================================================
// register index is out of range

s_add_i32 s106, s0, s1
// CHECK: error: register index is out of range
// CHECK-NEXT:{{^}}s_add_i32 s106, s0, s1
// CHECK-NEXT:{{^}}          ^

s_load_dwordx16 s[100:115], s[2:3], s4
// CHECK: error: register index is out of range
// CHECK-NEXT:{{^}}s_load_dwordx16 s[100:115], s[2:3], s4
// CHECK-NEXT:{{^}}                ^

s_mov_b32 ttmp16, 0
// CHECK: error: register index is out of range
// CHECK-NEXT:{{^}}s_mov_b32 ttmp16, 0
// CHECK-NEXT:{{^}}          ^

v_add_nc_i32 v256, v0, v1
// CHECK: error: register index is out of range
// CHECK-NEXT:{{^}}v_add_nc_i32 v256, v0, v1
// CHECK-NEXT:{{^}}             ^

//==============================================================================
// register not available on this GPU

s_and_b32     ttmp9, tma_hi, 0x0000ffff
// CHECK: error: register not available on this GPU
// CHECK-NEXT:{{^}}s_and_b32     ttmp9, tma_hi, 0x0000ffff
// CHECK-NEXT:{{^}}                     ^

s_mov_b32 flat_scratch, -1
// CHECK: error: register not available on this GPU
// CHECK-NEXT:{{^}}s_mov_b32 flat_scratch, -1
// CHECK-NEXT:{{^}}          ^

//==============================================================================
// registers in a list must be of the same kind

s_mov_b64 s[10:11], [a0,v1]
// CHECK: error: registers in a list must be of the same kind
// CHECK-NEXT:{{^}}s_mov_b64 s[10:11], [a0,v1]
// CHECK-NEXT:{{^}}                        ^

//==============================================================================
// registers in a list must have consecutive indices

s_mov_b64 s[10:11], [a0,a2]
// CHECK: error: registers in a list must have consecutive indices
// CHECK-NEXT:{{^}}s_mov_b64 s[10:11], [a0,a2]
// CHECK-NEXT:{{^}}                        ^

s_mov_b64 s[10:11], [s0,s0]
// CHECK: error: registers in a list must have consecutive indices
// CHECK-NEXT:{{^}}s_mov_b64 s[10:11], [s0,s0]
// CHECK-NEXT:{{^}}                        ^

s_mov_b64 s[10:11], [s2,s1]
// CHECK: error: registers in a list must have consecutive indices
// CHECK-NEXT:{{^}}s_mov_b64 s[10:11], [s2,s1]
// CHECK-NEXT:{{^}}                        ^

//==============================================================================
// source operand must be a VGPR

v_movrels_b32_sdwa v0, 1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: error: source operand must be a VGPR
// CHECK-NEXT:{{^}}v_movrels_b32_sdwa v0, 1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK-NEXT:{{^}}                       ^

v_movrels_b32_sdwa v0, s0
// CHECK: error: source operand must be a VGPR
// CHECK-NEXT:{{^}}v_movrels_b32_sdwa v0, s0
// CHECK-NEXT:{{^}}                       ^

v_movrels_b32_sdwa v0, shared_base
// CHECK: error: source operand must be a VGPR
// CHECK-NEXT:{{^}}v_movrels_b32_sdwa v0, shared_base
// CHECK-NEXT:{{^}}                       ^

//==============================================================================
// specified hardware register is not supported on this GPU

s_getreg_b32 s2, hwreg(HW_REG_SHADER_CYCLES)
// CHECK: error: specified hardware register is not supported on this GPU
// CHECK-NEXT:{{^}}s_getreg_b32 s2, hwreg(HW_REG_SHADER_CYCLES)
// CHECK-NEXT:{{^}}                       ^

//==============================================================================
// too few operands for instruction

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7]
// CHECK: error: too few operands for instruction
// CHECK-NEXT:{{^}}tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7]
// CHECK-NEXT:{{^}}^

v_add_f32_e64 v0, v1
// CHECK: error: too few operands for instruction
// CHECK-NEXT:{{^}}v_add_f32_e64 v0, v1
// CHECK-NEXT:{{^}}^

//==============================================================================
// too large value for expcnt

s_waitcnt expcnt(8)
// CHECK: error: too large value for expcnt
// CHECK-NEXT:{{^}}s_waitcnt expcnt(8)
// CHECK-NEXT:{{^}}                 ^

//==============================================================================
// too large value for lgkmcnt

s_waitcnt lgkmcnt(64)
// CHECK: error: too large value for lgkmcnt
// CHECK-NEXT:{{^}}s_waitcnt lgkmcnt(64)
// CHECK-NEXT:{{^}}                  ^

//==============================================================================
// too large value for vmcnt

s_waitcnt vmcnt(64)
// CHECK: error: too large value for vmcnt
// CHECK-NEXT:{{^}}s_waitcnt vmcnt(64)
// CHECK-NEXT:{{^}}                ^

//==============================================================================
// unknown token in expression

ds_swizzle_b32 v8, v2 offset:
// CHECK: error: unknown token in expression
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset:
// CHECK-NEXT:{{^}}                             ^

s_sendmsg sendmsg(1 -)
// CHECK: error: unknown token in expression
// CHECK-NEXT:{{^}}s_sendmsg sendmsg(1 -)
// CHECK-NEXT:{{^}}                     ^

tbuffer_load_format_d16_x v0, off, s[0:3], format:1,, s0
// CHECK: error: unknown token in expression
// CHECK-NEXT:{{^}}tbuffer_load_format_d16_x v0, off, s[0:3], format:1,, s0
// CHECK-NEXT:{{^}}                                                    ^

tbuffer_load_format_d16_x v0, off, s[0:3], format:1:, s0
// CHECK: error: unknown token in expression
// CHECK-NEXT:{{^}}tbuffer_load_format_d16_x v0, off, s[0:3], format:1:, s0
// CHECK-NEXT:{{^}}                                                   ^

v_pk_add_u16 v1, v2, v3 op_sel:[
// CHECK: error: unknown token in expression
// CHECK-NEXT:{{^}}v_pk_add_u16 v1, v2, v3 op_sel:[
// CHECK-NEXT:{{^}}                                ^

v_pk_add_u16 v1, v2, v3 op_sel:[,0]
// CHECK: error: unknown token in expression
// CHECK-NEXT:{{^}}v_pk_add_u16 v1, v2, v3 op_sel:[,0]
// CHECK-NEXT:{{^}}                                ^

v_pk_add_u16 v1, v2, v3 op_sel:[,]
// CHECK: error: unknown token in expression
// CHECK-NEXT:{{^}}v_pk_add_u16 v1, v2, v3 op_sel:[,]
// CHECK-NEXT:{{^}}                                ^

v_pk_add_u16 v1, v2, v3 op_sel:[0,]
// CHECK: error: unknown token in expression
// CHECK-NEXT:{{^}}v_pk_add_u16 v1, v2, v3 op_sel:[0,]
// CHECK-NEXT:{{^}}                                  ^

v_pk_add_u16 v1, v2, v3 op_sel:[]
// CHECK: error: unknown token in expression
// CHECK-NEXT:{{^}}v_pk_add_u16 v1, v2, v3 op_sel:[]
// CHECK-NEXT:{{^}}                                ^

//==============================================================================
// unsupported format

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT]
// CHECK: error: unsupported format
// CHECK-NEXT:{{^}}tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT]
// CHECK-NEXT:{{^}}                                                             ^

//==============================================================================
// expected vertical bar

v_ceil_f32 v1, |1+1|
// CHECK: error: expected vertical bar
// CHECK-NEXT:{{^}}v_ceil_f32 v1, |1+1|
// CHECK-NEXT:{{^}}                 ^

//==============================================================================
// expected left paren after neg

v_ceil_f32 v1, neg-(v2)
// CHECK: error: expected left paren after neg
// CHECK-NEXT:{{^}}v_ceil_f32 v1, neg-(v2)
// CHECK-NEXT:{{^}}                  ^

//==============================================================================
// expected left paren after abs

v_ceil_f32 v1, abs-(v2)
// CHECK: error: expected left paren after abs
// CHECK-NEXT:{{^}}v_ceil_f32 v1, abs-(v2)
// CHECK-NEXT:{{^}}                  ^

//==============================================================================
// expected left paren after sext

v_cmpx_f_i32_sdwa sext[v1], v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: error: expected left paren after sext
// CHECK-NEXT:{{^}}v_cmpx_f_i32_sdwa sext[v1], v2 src0_sel:DWORD src1_sel:DWORD
// CHECK-NEXT:{{^}}                      ^

//==============================================================================
// expected closing parentheses

v_ceil_f32 v1, abs(v2]
// CHECK: error: expected closing parentheses
// CHECK-NEXT:{{^}}v_ceil_f32 v1, abs(v2]
// CHECK-NEXT:{{^}}                     ^

v_ceil_f32 v1, neg(v2]
// CHECK: error: expected closing parentheses
// CHECK-NEXT:{{^}}v_ceil_f32 v1, neg(v2]
// CHECK-NEXT:{{^}}                     ^

v_cmpx_f_i32_sdwa sext(v1], v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: error: expected closing parentheses
// CHECK-NEXT:{{^}}v_cmpx_f_i32_sdwa sext(v1], v2 src0_sel:DWORD src1_sel:DWORD
// CHECK-NEXT:{{^}}                         ^

//==============================================================================
// expected a left parentheses

ds_swizzle_b32 v8, v2 offset:swizzle[QUAD_PERM, 0, 1, 2, 3]
// CHECK: error: expected a left parentheses
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset:swizzle[QUAD_PERM, 0, 1, 2, 3]
// CHECK-NEXT:{{^}}                                    ^

//==============================================================================
// expected an absolute expression or a label

s_branch 1+x
// CHECK: error: expected an absolute expression or a label
// CHECK-NEXT:{{^}}s_branch 1+x
// CHECK-NEXT:{{^}}         ^

//==============================================================================
// expected a 16-bit offset

ds_swizzle_b32 v8, v2 offset:0x10000
// CHECK: error: expected a 16-bit offset
// CHECK-NEXT:{{^}}ds_swizzle_b32 v8, v2 offset:0x10000
// CHECK-NEXT:{{^}}                             ^
