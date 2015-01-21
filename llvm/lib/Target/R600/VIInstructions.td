//===-- VIInstructions.td - VI Instruction Defintions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Instruction definitions for VI and newer.
//===----------------------------------------------------------------------===//

let SubtargetPredicate = isVI in {

defm BUFFER_LOAD_DWORD_VI : MUBUF_Load_Helper_vi <
  0x14, "buffer_load_dword", VGPR_32, i32, global_load
>;

defm BUFFER_LOAD_FORMAT_XYZW_VI : MUBUF_Load_Helper_vi <
  0x03, "buffer_load_format_xyzw", VReg_128
>;

} // End SubtargetPredicate = isVI


//===----------------------------------------------------------------------===//
// SMEM Patterns
//===----------------------------------------------------------------------===//

let Predicates = [isVI] in {

// 1. Offset as 8bit DWORD immediate
def : Pat <
  (SIload_constant v4i32:$sbase, IMM20bit:$offset),
  (S_BUFFER_LOAD_DWORD_IMM $sbase, (as_i32imm $offset))
>;

//===----------------------------------------------------------------------===//
// MUBUF Patterns
//===----------------------------------------------------------------------===//

// Offset in an 32Bit VGPR
def : Pat <
  (SIload_constant v4i32:$sbase, i32:$voff),
  (BUFFER_LOAD_DWORD_VI_OFFEN $sbase, $voff, 0, 0, 0, 0, 0)
>;

// Offset in an 32Bit VGPR
def : Pat <
  (SIload_constant v4i32:$sbase, i32:$voff),
  (BUFFER_LOAD_DWORD_VI_OFFEN $sbase, $voff, 0, 0, 0, 0, 0)
>;

/* int_SI_vs_load_input */
def : Pat<
  (SIload_input v4i32:$tlst, imm:$attr_offset, i32:$buf_idx_vgpr),
  (BUFFER_LOAD_FORMAT_XYZW_VI_IDXEN $tlst, $buf_idx_vgpr, imm:$attr_offset, 0, 0, 0, 0)
>;

defm : MUBUF_Load_Dword <i32, BUFFER_LOAD_DWORD_VI_OFFSET,
                         BUFFER_LOAD_DWORD_VI_OFFEN,
                         BUFFER_LOAD_DWORD_VI_IDXEN,
                         BUFFER_LOAD_DWORD_VI_BOTHEN>;

} // End Predicates = [isVI]
