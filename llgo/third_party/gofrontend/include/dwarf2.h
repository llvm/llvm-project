//===----------------------------- dwarf2.h -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
// DWARF constants. Derived from:
//   - libcxxabi/src/Unwind/dwarf2.h
//   - DWARF 4 specification
//
//===----------------------------------------------------------------------===//

#ifndef DWARF2_H
#define DWARF2_H

enum dwarf_attribute {
  DW_AT_name = 0x03,
  DW_AT_stmt_list = 0x10,
  DW_AT_low_pc = 0x11,
  DW_AT_high_pc = 0x12,
  DW_AT_comp_dir = 0x1b,
  DW_AT_abstract_origin = 0x31,
  DW_AT_specification = 0x47,
  DW_AT_ranges = 0x55,
  DW_AT_call_file = 0x58,
  DW_AT_call_line = 0x59,
  DW_AT_linkage_name = 0x6e,
  DW_AT_MIPS_linkage_name = 0x2007
};

enum dwarf_form {
  DW_FORM_addr = 0x01,
  DW_FORM_block2 = 0x03,
  DW_FORM_block4 = 0x04,
  DW_FORM_data2 = 0x05,
  DW_FORM_data4 = 0x06,
  DW_FORM_data8 = 0x07,
  DW_FORM_string = 0x08,
  DW_FORM_block = 0x09,
  DW_FORM_block1 = 0x0a,
  DW_FORM_data1 = 0x0b,
  DW_FORM_flag = 0x0c,
  DW_FORM_sdata = 0x0d,
  DW_FORM_strp = 0x0e,
  DW_FORM_udata = 0x0f,
  DW_FORM_ref_addr = 0x10,
  DW_FORM_ref1 = 0x11,
  DW_FORM_ref2 = 0x12,
  DW_FORM_ref4 = 0x13,
  DW_FORM_ref8 = 0x14,
  DW_FORM_ref_udata = 0x15,
  DW_FORM_indirect = 0x16,
  DW_FORM_sec_offset = 0x17,
  DW_FORM_exprloc = 0x18,
  DW_FORM_flag_present = 0x19,
  DW_FORM_ref_sig8 = 0x20,
  DW_FORM_GNU_addr_index = 0x1f01,
  DW_FORM_GNU_str_index = 0x1f02,
  DW_FORM_GNU_ref_alt = 0x1f20,
  DW_FORM_GNU_strp_alt = 0x1f21
};

enum dwarf_tag {
  DW_TAG_entry_point = 0x03,
  DW_TAG_compile_unit = 0x11,
  DW_TAG_inlined_subroutine = 0x1d,
  DW_TAG_subprogram = 0x2e
};

enum dwarf_lns {
  DW_LNS_extended_op = 0x00,
  DW_LNS_copy = 0x01,
  DW_LNS_advance_pc = 0x02,
  DW_LNS_advance_line = 0x03,
  DW_LNS_set_file = 0x04,
  DW_LNS_set_column = 0x05,
  DW_LNS_negate_stmt = 0x06,
  DW_LNS_set_basic_block = 0x07,
  DW_LNS_const_add_pc = 0x08,
  DW_LNS_fixed_advance_pc = 0x09,
  DW_LNS_set_prologue_end = 0x0a,
  DW_LNS_set_epilogue_begin = 0x0b,
  DW_LNS_set_isa = 0x0c
};

enum dwarf_lne {
  DW_LNE_end_sequence = 0x01,
  DW_LNE_set_address = 0x02,
  DW_LNE_define_file = 0x03,
  DW_LNE_set_discriminator = 0x04
};

#endif  // DWARF2_H
