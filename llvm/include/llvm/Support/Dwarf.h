//===-- llvm/Support/Dwarf.h ---Dwarf Constants------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains constants used for implementing Dwarf debug support.  For
// Details on the Dwarf 3 specfication see DWARF Debugging Information Format
// V.3 reference manual http://dwarf.freestandards.org ,
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DWARF_H
#define LLVM_SUPPORT_DWARF_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

//===----------------------------------------------------------------------===//
// Debug info constants.

enum LLVM_ENUM_INT_TYPE(uint32_t) {
  LLVMDebugVersion = (12 << 16),    // Current version of debug information.
  LLVMDebugVersion11 = (11 << 16),  // Constant for version 11.
  LLVMDebugVersion10 = (10 << 16),  // Constant for version 10.
  LLVMDebugVersion9 = (9 << 16),    // Constant for version 9.
  LLVMDebugVersion8 = (8 << 16),    // Constant for version 8.
  LLVMDebugVersion7 = (7 << 16),    // Constant for version 7.
  LLVMDebugVersion6 = (6 << 16),    // Constant for version 6.
  LLVMDebugVersion5 = (5 << 16),    // Constant for version 5.
  LLVMDebugVersion4 = (4 << 16),    // Constant for version 4.
  LLVMDebugVersionMask = 0xffff0000 // Mask for version number.
};

namespace dwarf {

//===----------------------------------------------------------------------===//
// Dwarf constants as gleaned from the DWARF Debugging Information Format V.4
// reference manual http://dwarf.freestandards.org.
//

// Do not mix the following two enumerations sets.  DW_TAG_invalid changes the
// enumeration base type.

enum LLVMConstants LLVM_ENUM_INT_TYPE(uint32_t) {
  // llvm mock tags
  DW_TAG_invalid = ~0U, // Tag for invalid results.

  DW_TAG_auto_variable = 0x100, // Tag for local (auto) variables.
  DW_TAG_arg_variable = 0x101,  // Tag for argument variables.

  DW_TAG_user_base = 0x1000, // Recommended base for user tags.

  DWARF_VERSION = 4,       // Default dwarf version we output.
  DW_CIE_VERSION = 1,      // Common frame information version.
  DW_PUBTYPES_VERSION = 2, // Section version number for .debug_pubtypes.
  DW_PUBNAMES_VERSION = 2, // Section version number for .debug_pubnames.
  DW_ARANGES_VERSION = 2   // Section version number for .debug_aranges.
};

// Special ID values that distinguish a CIE from a FDE in DWARF CFI.
// Not inside an enum because a 64-bit value is needed.
const uint32_t DW_CIE_ID = UINT32_MAX;
const uint64_t DW64_CIE_ID = UINT64_MAX;

enum Tag LLVM_ENUM_INT_TYPE(uint16_t) {
  DW_TAG_array_type = 0x01,
  DW_TAG_class_type = 0x02,
  DW_TAG_entry_point = 0x03,
  DW_TAG_enumeration_type = 0x04,
  DW_TAG_formal_parameter = 0x05,
  DW_TAG_imported_declaration = 0x08,
  DW_TAG_label = 0x0a,
  DW_TAG_lexical_block = 0x0b,
  DW_TAG_member = 0x0d,
  DW_TAG_pointer_type = 0x0f,
  DW_TAG_reference_type = 0x10,
  DW_TAG_compile_unit = 0x11,
  DW_TAG_string_type = 0x12,
  DW_TAG_structure_type = 0x13,
  DW_TAG_subroutine_type = 0x15,
  DW_TAG_typedef = 0x16,
  DW_TAG_union_type = 0x17,
  DW_TAG_unspecified_parameters = 0x18,
  DW_TAG_variant = 0x19,
  DW_TAG_common_block = 0x1a,
  DW_TAG_common_inclusion = 0x1b,
  DW_TAG_inheritance = 0x1c,
  DW_TAG_inlined_subroutine = 0x1d,
  DW_TAG_module = 0x1e,
  DW_TAG_ptr_to_member_type = 0x1f,
  DW_TAG_set_type = 0x20,
  DW_TAG_subrange_type = 0x21,
  DW_TAG_with_stmt = 0x22,
  DW_TAG_access_declaration = 0x23,
  DW_TAG_base_type = 0x24,
  DW_TAG_catch_block = 0x25,
  DW_TAG_const_type = 0x26,
  DW_TAG_constant = 0x27,
  DW_TAG_enumerator = 0x28,
  DW_TAG_file_type = 0x29,
  DW_TAG_friend = 0x2a,
  DW_TAG_namelist = 0x2b,
  DW_TAG_namelist_item = 0x2c,
  DW_TAG_packed_type = 0x2d,
  DW_TAG_subprogram = 0x2e,
  DW_TAG_template_type_parameter = 0x2f,
  DW_TAG_template_value_parameter = 0x30,
  DW_TAG_thrown_type = 0x31,
  DW_TAG_try_block = 0x32,
  DW_TAG_variant_part = 0x33,
  DW_TAG_variable = 0x34,
  DW_TAG_volatile_type = 0x35,
  DW_TAG_dwarf_procedure = 0x36,
  DW_TAG_restrict_type = 0x37,
  DW_TAG_interface_type = 0x38,
  DW_TAG_namespace = 0x39,
  DW_TAG_imported_module = 0x3a,
  DW_TAG_unspecified_type = 0x3b,
  DW_TAG_partial_unit = 0x3c,
  DW_TAG_imported_unit = 0x3d,
  DW_TAG_condition = 0x3f,
  DW_TAG_shared_type = 0x40,
  DW_TAG_type_unit = 0x41,
  DW_TAG_rvalue_reference_type = 0x42,
  DW_TAG_template_alias = 0x43,
  DW_TAG_MIPS_loop = 0x4081,
  DW_TAG_format_label = 0x4101,
  DW_TAG_function_template = 0x4102,
  DW_TAG_class_template = 0x4103,
  DW_TAG_GNU_template_template_param = 0x4106,
  DW_TAG_GNU_template_parameter_pack = 0x4107,
  DW_TAG_GNU_formal_parameter_pack = 0x4108,
  DW_TAG_lo_user = 0x4080,
  DW_TAG_APPLE_property = 0x4200,
  DW_TAG_hi_user = 0xffff
};

inline bool isType(Tag T) {
  switch (T) {
  case DW_TAG_array_type:
  case DW_TAG_class_type:
  case DW_TAG_interface_type:
  case DW_TAG_enumeration_type:
  case DW_TAG_pointer_type:
  case DW_TAG_reference_type:
  case DW_TAG_rvalue_reference_type:
  case DW_TAG_string_type:
  case DW_TAG_structure_type:
  case DW_TAG_subroutine_type:
  case DW_TAG_union_type:
  case DW_TAG_ptr_to_member_type:
  case DW_TAG_set_type:
  case DW_TAG_subrange_type:
  case DW_TAG_base_type:
  case DW_TAG_const_type:
  case DW_TAG_file_type:
  case DW_TAG_packed_type:
  case DW_TAG_volatile_type:
  case DW_TAG_typedef:
    return true;
  default:
    return false;
  }
}

enum Attribute LLVM_ENUM_INT_TYPE(uint16_t) {
  // Attributes
  DW_AT_sibling = 0x01,
  DW_AT_location = 0x02,
  DW_AT_name = 0x03,
  DW_AT_ordering = 0x09,
  DW_AT_byte_size = 0x0b,
  DW_AT_bit_offset = 0x0c,
  DW_AT_bit_size = 0x0d,
  DW_AT_stmt_list = 0x10,
  DW_AT_low_pc = 0x11,
  DW_AT_high_pc = 0x12,
  DW_AT_language = 0x13,
  DW_AT_discr = 0x15,
  DW_AT_discr_value = 0x16,
  DW_AT_visibility = 0x17,
  DW_AT_import = 0x18,
  DW_AT_string_length = 0x19,
  DW_AT_common_reference = 0x1a,
  DW_AT_comp_dir = 0x1b,
  DW_AT_const_value = 0x1c,
  DW_AT_containing_type = 0x1d,
  DW_AT_default_value = 0x1e,
  DW_AT_inline = 0x20,
  DW_AT_is_optional = 0x21,
  DW_AT_lower_bound = 0x22,
  DW_AT_producer = 0x25,
  DW_AT_prototyped = 0x27,
  DW_AT_return_addr = 0x2a,
  DW_AT_start_scope = 0x2c,
  DW_AT_bit_stride = 0x2e,
  DW_AT_upper_bound = 0x2f,
  DW_AT_abstract_origin = 0x31,
  DW_AT_accessibility = 0x32,
  DW_AT_address_class = 0x33,
  DW_AT_artificial = 0x34,
  DW_AT_base_types = 0x35,
  DW_AT_calling_convention = 0x36,
  DW_AT_count = 0x37,
  DW_AT_data_member_location = 0x38,
  DW_AT_decl_column = 0x39,
  DW_AT_decl_file = 0x3a,
  DW_AT_decl_line = 0x3b,
  DW_AT_declaration = 0x3c,
  DW_AT_discr_list = 0x3d,
  DW_AT_encoding = 0x3e,
  DW_AT_external = 0x3f,
  DW_AT_frame_base = 0x40,
  DW_AT_friend = 0x41,
  DW_AT_identifier_case = 0x42,
  DW_AT_macro_info = 0x43,
  DW_AT_namelist_item = 0x44,
  DW_AT_priority = 0x45,
  DW_AT_segment = 0x46,
  DW_AT_specification = 0x47,
  DW_AT_static_link = 0x48,
  DW_AT_type = 0x49,
  DW_AT_use_location = 0x4a,
  DW_AT_variable_parameter = 0x4b,
  DW_AT_virtuality = 0x4c,
  DW_AT_vtable_elem_location = 0x4d,
  DW_AT_allocated = 0x4e,
  DW_AT_associated = 0x4f,
  DW_AT_data_location = 0x50,
  DW_AT_byte_stride = 0x51,
  DW_AT_entry_pc = 0x52,
  DW_AT_use_UTF8 = 0x53,
  DW_AT_extension = 0x54,
  DW_AT_ranges = 0x55,
  DW_AT_trampoline = 0x56,
  DW_AT_call_column = 0x57,
  DW_AT_call_file = 0x58,
  DW_AT_call_line = 0x59,
  DW_AT_description = 0x5a,
  DW_AT_binary_scale = 0x5b,
  DW_AT_decimal_scale = 0x5c,
  DW_AT_small = 0x5d,
  DW_AT_decimal_sign = 0x5e,
  DW_AT_digit_count = 0x5f,
  DW_AT_picture_string = 0x60,
  DW_AT_mutable = 0x61,
  DW_AT_threads_scaled = 0x62,
  DW_AT_explicit = 0x63,
  DW_AT_object_pointer = 0x64,
  DW_AT_endianity = 0x65,
  DW_AT_elemental = 0x66,
  DW_AT_pure = 0x67,
  DW_AT_recursive = 0x68,
  DW_AT_signature = 0x69,
  DW_AT_main_subprogram = 0x6a,
  DW_AT_data_bit_offset = 0x6b,
  DW_AT_const_expr = 0x6c,
  DW_AT_enum_class = 0x6d,
  DW_AT_linkage_name = 0x6e,

  DW_AT_lo_user = 0x2000,
  DW_AT_hi_user = 0x3fff,

  DW_AT_MIPS_loop_begin = 0x2002,
  DW_AT_MIPS_tail_loop_begin = 0x2003,
  DW_AT_MIPS_epilog_begin = 0x2004,
  DW_AT_MIPS_loop_unroll_factor = 0x2005,
  DW_AT_MIPS_software_pipeline_depth = 0x2006,
  DW_AT_MIPS_linkage_name = 0x2007,
  DW_AT_MIPS_stride = 0x2008,
  DW_AT_MIPS_abstract_name = 0x2009,
  DW_AT_MIPS_clone_origin = 0x200a,
  DW_AT_MIPS_has_inlines = 0x200b,
  DW_AT_MIPS_stride_byte = 0x200c,
  DW_AT_MIPS_stride_elem = 0x200d,
  DW_AT_MIPS_ptr_dopetype = 0x200e,
  DW_AT_MIPS_allocatable_dopetype = 0x200f,
  DW_AT_MIPS_assumed_shape_dopetype = 0x2010,

  // This one appears to have only been implemented by Open64 for
  // fortran and may conflict with other extensions.
  DW_AT_MIPS_assumed_size = 0x2011,

  // GNU extensions
  DW_AT_sf_names = 0x2101,
  DW_AT_src_info = 0x2102,
  DW_AT_mac_info = 0x2103,
  DW_AT_src_coords = 0x2104,
  DW_AT_body_begin = 0x2105,
  DW_AT_body_end = 0x2106,
  DW_AT_GNU_vector = 0x2107,
  DW_AT_GNU_template_name = 0x2110,

  DW_AT_GNU_odr_signature = 0x210f,

  // Extensions for Fission proposal.
  DW_AT_GNU_dwo_name = 0x2130,
  DW_AT_GNU_dwo_id = 0x2131,
  DW_AT_GNU_ranges_base = 0x2132,
  DW_AT_GNU_addr_base = 0x2133,
  DW_AT_GNU_pubnames = 0x2134,
  DW_AT_GNU_pubtypes = 0x2135,

  // Apple extensions.
  DW_AT_APPLE_optimized = 0x3fe1,
  DW_AT_APPLE_flags = 0x3fe2,
  DW_AT_APPLE_isa = 0x3fe3,
  DW_AT_APPLE_block = 0x3fe4,
  DW_AT_APPLE_major_runtime_vers = 0x3fe5,
  DW_AT_APPLE_runtime_class = 0x3fe6,
  DW_AT_APPLE_omit_frame_ptr = 0x3fe7,
  DW_AT_APPLE_property_name = 0x3fe8,
  DW_AT_APPLE_property_getter = 0x3fe9,
  DW_AT_APPLE_property_setter = 0x3fea,
  DW_AT_APPLE_property_attribute = 0x3feb,
  DW_AT_APPLE_objc_complete_type = 0x3fec,
  DW_AT_APPLE_property = 0x3fed
};

enum Form LLVM_ENUM_INT_TYPE(uint16_t) {
  // Attribute form encodings
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

  // Extensions for Fission proposal
  DW_FORM_GNU_addr_index = 0x1f01,
  DW_FORM_GNU_str_index = 0x1f02
};

enum LocationAtom {
  // Operation encodings
  DW_OP_addr = 0x03,
  DW_OP_deref = 0x06,
  DW_OP_const1u = 0x08,
  DW_OP_const1s = 0x09,
  DW_OP_const2u = 0x0a,
  DW_OP_const2s = 0x0b,
  DW_OP_const4u = 0x0c,
  DW_OP_const4s = 0x0d,
  DW_OP_const8u = 0x0e,
  DW_OP_const8s = 0x0f,
  DW_OP_constu = 0x10,
  DW_OP_consts = 0x11,
  DW_OP_dup = 0x12,
  DW_OP_drop = 0x13,
  DW_OP_over = 0x14,
  DW_OP_pick = 0x15,
  DW_OP_swap = 0x16,
  DW_OP_rot = 0x17,
  DW_OP_xderef = 0x18,
  DW_OP_abs = 0x19,
  DW_OP_and = 0x1a,
  DW_OP_div = 0x1b,
  DW_OP_minus = 0x1c,
  DW_OP_mod = 0x1d,
  DW_OP_mul = 0x1e,
  DW_OP_neg = 0x1f,
  DW_OP_not = 0x20,
  DW_OP_or = 0x21,
  DW_OP_plus = 0x22,
  DW_OP_plus_uconst = 0x23,
  DW_OP_shl = 0x24,
  DW_OP_shr = 0x25,
  DW_OP_shra = 0x26,
  DW_OP_xor = 0x27,
  DW_OP_skip = 0x2f,
  DW_OP_bra = 0x28,
  DW_OP_eq = 0x29,
  DW_OP_ge = 0x2a,
  DW_OP_gt = 0x2b,
  DW_OP_le = 0x2c,
  DW_OP_lt = 0x2d,
  DW_OP_ne = 0x2e,
  DW_OP_lit0 = 0x30,
  DW_OP_lit1 = 0x31,
  DW_OP_lit2 = 0x32,
  DW_OP_lit3 = 0x33,
  DW_OP_lit4 = 0x34,
  DW_OP_lit5 = 0x35,
  DW_OP_lit6 = 0x36,
  DW_OP_lit7 = 0x37,
  DW_OP_lit8 = 0x38,
  DW_OP_lit9 = 0x39,
  DW_OP_lit10 = 0x3a,
  DW_OP_lit11 = 0x3b,
  DW_OP_lit12 = 0x3c,
  DW_OP_lit13 = 0x3d,
  DW_OP_lit14 = 0x3e,
  DW_OP_lit15 = 0x3f,
  DW_OP_lit16 = 0x40,
  DW_OP_lit17 = 0x41,
  DW_OP_lit18 = 0x42,
  DW_OP_lit19 = 0x43,
  DW_OP_lit20 = 0x44,
  DW_OP_lit21 = 0x45,
  DW_OP_lit22 = 0x46,
  DW_OP_lit23 = 0x47,
  DW_OP_lit24 = 0x48,
  DW_OP_lit25 = 0x49,
  DW_OP_lit26 = 0x4a,
  DW_OP_lit27 = 0x4b,
  DW_OP_lit28 = 0x4c,
  DW_OP_lit29 = 0x4d,
  DW_OP_lit30 = 0x4e,
  DW_OP_lit31 = 0x4f,
  DW_OP_reg0 = 0x50,
  DW_OP_reg1 = 0x51,
  DW_OP_reg2 = 0x52,
  DW_OP_reg3 = 0x53,
  DW_OP_reg4 = 0x54,
  DW_OP_reg5 = 0x55,
  DW_OP_reg6 = 0x56,
  DW_OP_reg7 = 0x57,
  DW_OP_reg8 = 0x58,
  DW_OP_reg9 = 0x59,
  DW_OP_reg10 = 0x5a,
  DW_OP_reg11 = 0x5b,
  DW_OP_reg12 = 0x5c,
  DW_OP_reg13 = 0x5d,
  DW_OP_reg14 = 0x5e,
  DW_OP_reg15 = 0x5f,
  DW_OP_reg16 = 0x60,
  DW_OP_reg17 = 0x61,
  DW_OP_reg18 = 0x62,
  DW_OP_reg19 = 0x63,
  DW_OP_reg20 = 0x64,
  DW_OP_reg21 = 0x65,
  DW_OP_reg22 = 0x66,
  DW_OP_reg23 = 0x67,
  DW_OP_reg24 = 0x68,
  DW_OP_reg25 = 0x69,
  DW_OP_reg26 = 0x6a,
  DW_OP_reg27 = 0x6b,
  DW_OP_reg28 = 0x6c,
  DW_OP_reg29 = 0x6d,
  DW_OP_reg30 = 0x6e,
  DW_OP_reg31 = 0x6f,
  DW_OP_breg0 = 0x70,
  DW_OP_breg1 = 0x71,
  DW_OP_breg2 = 0x72,
  DW_OP_breg3 = 0x73,
  DW_OP_breg4 = 0x74,
  DW_OP_breg5 = 0x75,
  DW_OP_breg6 = 0x76,
  DW_OP_breg7 = 0x77,
  DW_OP_breg8 = 0x78,
  DW_OP_breg9 = 0x79,
  DW_OP_breg10 = 0x7a,
  DW_OP_breg11 = 0x7b,
  DW_OP_breg12 = 0x7c,
  DW_OP_breg13 = 0x7d,
  DW_OP_breg14 = 0x7e,
  DW_OP_breg15 = 0x7f,
  DW_OP_breg16 = 0x80,
  DW_OP_breg17 = 0x81,
  DW_OP_breg18 = 0x82,
  DW_OP_breg19 = 0x83,
  DW_OP_breg20 = 0x84,
  DW_OP_breg21 = 0x85,
  DW_OP_breg22 = 0x86,
  DW_OP_breg23 = 0x87,
  DW_OP_breg24 = 0x88,
  DW_OP_breg25 = 0x89,
  DW_OP_breg26 = 0x8a,
  DW_OP_breg27 = 0x8b,
  DW_OP_breg28 = 0x8c,
  DW_OP_breg29 = 0x8d,
  DW_OP_breg30 = 0x8e,
  DW_OP_breg31 = 0x8f,
  DW_OP_regx = 0x90,
  DW_OP_fbreg = 0x91,
  DW_OP_bregx = 0x92,
  DW_OP_piece = 0x93,
  DW_OP_deref_size = 0x94,
  DW_OP_xderef_size = 0x95,
  DW_OP_nop = 0x96,
  DW_OP_push_object_address = 0x97,
  DW_OP_call2 = 0x98,
  DW_OP_call4 = 0x99,
  DW_OP_call_ref = 0x9a,
  DW_OP_form_tls_address = 0x9b,
  DW_OP_call_frame_cfa = 0x9c,
  DW_OP_bit_piece = 0x9d,
  DW_OP_implicit_value = 0x9e,
  DW_OP_stack_value = 0x9f,
  DW_OP_lo_user = 0xe0,
  DW_OP_hi_user = 0xff,

  // Extensions for GNU-style thread-local storage.
  DW_OP_GNU_push_tls_address = 0xe0,

  // Extensions for Fission proposal.
  DW_OP_GNU_addr_index = 0xfb,
  DW_OP_GNU_const_index = 0xfc
};

enum TypeKind {
  // Encoding attribute values
  DW_ATE_address = 0x01,
  DW_ATE_boolean = 0x02,
  DW_ATE_complex_float = 0x03,
  DW_ATE_float = 0x04,
  DW_ATE_signed = 0x05,
  DW_ATE_signed_char = 0x06,
  DW_ATE_unsigned = 0x07,
  DW_ATE_unsigned_char = 0x08,
  DW_ATE_imaginary_float = 0x09,
  DW_ATE_packed_decimal = 0x0a,
  DW_ATE_numeric_string = 0x0b,
  DW_ATE_edited = 0x0c,
  DW_ATE_signed_fixed = 0x0d,
  DW_ATE_unsigned_fixed = 0x0e,
  DW_ATE_decimal_float = 0x0f,
  DW_ATE_UTF = 0x10,
  DW_ATE_lo_user = 0x80,
  DW_ATE_hi_user = 0xff
};

enum DecimalSignEncoding {
  // Decimal sign attribute values
  DW_DS_unsigned = 0x01,
  DW_DS_leading_overpunch = 0x02,
  DW_DS_trailing_overpunch = 0x03,
  DW_DS_leading_separate = 0x04,
  DW_DS_trailing_separate = 0x05
};

enum EndianityEncoding {
  // Endianity attribute values
  DW_END_default = 0x00,
  DW_END_big = 0x01,
  DW_END_little = 0x02,
  DW_END_lo_user = 0x40,
  DW_END_hi_user = 0xff
};

enum AccessAttribute {
  // Accessibility codes
  DW_ACCESS_public = 0x01,
  DW_ACCESS_protected = 0x02,
  DW_ACCESS_private = 0x03
};

enum VisibilityAttribute {
  // Visibility codes
  DW_VIS_local = 0x01,
  DW_VIS_exported = 0x02,
  DW_VIS_qualified = 0x03
};

enum VirtualityAttribute {
  // Virtuality codes
  DW_VIRTUALITY_none = 0x00,
  DW_VIRTUALITY_virtual = 0x01,
  DW_VIRTUALITY_pure_virtual = 0x02
};

enum SourceLanguage {
  // Language names
  DW_LANG_C89 = 0x0001,
  DW_LANG_C = 0x0002,
  DW_LANG_Ada83 = 0x0003,
  DW_LANG_C_plus_plus = 0x0004,
  DW_LANG_Cobol74 = 0x0005,
  DW_LANG_Cobol85 = 0x0006,
  DW_LANG_Fortran77 = 0x0007,
  DW_LANG_Fortran90 = 0x0008,
  DW_LANG_Pascal83 = 0x0009,
  DW_LANG_Modula2 = 0x000a,
  DW_LANG_Java = 0x000b,
  DW_LANG_C99 = 0x000c,
  DW_LANG_Ada95 = 0x000d,
  DW_LANG_Fortran95 = 0x000e,
  DW_LANG_PLI = 0x000f,
  DW_LANG_ObjC = 0x0010,
  DW_LANG_ObjC_plus_plus = 0x0011,
  DW_LANG_UPC = 0x0012,
  DW_LANG_D = 0x0013,
  DW_LANG_Python = 0x0014,
  DW_LANG_lo_user = 0x8000,
  DW_LANG_Mips_Assembler = 0x8001,
  DW_LANG_hi_user = 0xffff
};

enum CaseSensitivity {
  // Identifier case codes
  DW_ID_case_sensitive = 0x00,
  DW_ID_up_case = 0x01,
  DW_ID_down_case = 0x02,
  DW_ID_case_insensitive = 0x03
};

enum CallingConvention {
  // Calling convention codes
  DW_CC_normal = 0x01,
  DW_CC_program = 0x02,
  DW_CC_nocall = 0x03,
  DW_CC_lo_user = 0x40,
  DW_CC_hi_user = 0xff
};

enum InlineAttribute {
  // Inline codes
  DW_INL_not_inlined = 0x00,
  DW_INL_inlined = 0x01,
  DW_INL_declared_not_inlined = 0x02,
  DW_INL_declared_inlined = 0x03
};

enum ArrayDimensionOrdering {
  // Array ordering
  DW_ORD_row_major = 0x00,
  DW_ORD_col_major = 0x01
};

enum DiscriminantList {
  // Discriminant descriptor values
  DW_DSC_label = 0x00,
  DW_DSC_range = 0x01
};

enum LineNumberOps {
  // Line Number Standard Opcode Encodings
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

enum LineNumberExtendedOps {
  // Line Number Extended Opcode Encodings
  DW_LNE_end_sequence = 0x01,
  DW_LNE_set_address = 0x02,
  DW_LNE_define_file = 0x03,
  DW_LNE_set_discriminator = 0x04,
  DW_LNE_lo_user = 0x80,
  DW_LNE_hi_user = 0xff
};

enum MacinfoRecordType {
  // Macinfo Type Encodings
  DW_MACINFO_define = 0x01,
  DW_MACINFO_undef = 0x02,
  DW_MACINFO_start_file = 0x03,
  DW_MACINFO_end_file = 0x04,
  DW_MACINFO_vendor_ext = 0xff
};

enum CallFrameInfo {
  // Call frame instruction encodings
  DW_CFA_extended = 0x00,
  DW_CFA_nop = 0x00,
  DW_CFA_advance_loc = 0x40,
  DW_CFA_offset = 0x80,
  DW_CFA_restore = 0xc0,
  DW_CFA_set_loc = 0x01,
  DW_CFA_advance_loc1 = 0x02,
  DW_CFA_advance_loc2 = 0x03,
  DW_CFA_advance_loc4 = 0x04,
  DW_CFA_offset_extended = 0x05,
  DW_CFA_restore_extended = 0x06,
  DW_CFA_undefined = 0x07,
  DW_CFA_same_value = 0x08,
  DW_CFA_register = 0x09,
  DW_CFA_remember_state = 0x0a,
  DW_CFA_restore_state = 0x0b,
  DW_CFA_def_cfa = 0x0c,
  DW_CFA_def_cfa_register = 0x0d,
  DW_CFA_def_cfa_offset = 0x0e,
  DW_CFA_def_cfa_expression = 0x0f,
  DW_CFA_expression = 0x10,
  DW_CFA_offset_extended_sf = 0x11,
  DW_CFA_def_cfa_sf = 0x12,
  DW_CFA_def_cfa_offset_sf = 0x13,
  DW_CFA_val_offset = 0x14,
  DW_CFA_val_offset_sf = 0x15,
  DW_CFA_val_expression = 0x16,
  DW_CFA_MIPS_advance_loc8 = 0x1d,
  DW_CFA_GNU_window_save = 0x2d,
  DW_CFA_GNU_args_size = 0x2e,
  DW_CFA_lo_user = 0x1c,
  DW_CFA_hi_user = 0x3f
};

enum Constants {
  // Children flag
  DW_CHILDREN_no = 0x00,
  DW_CHILDREN_yes = 0x01,

  DW_EH_PE_absptr = 0x00,
  DW_EH_PE_omit = 0xff,
  DW_EH_PE_uleb128 = 0x01,
  DW_EH_PE_udata2 = 0x02,
  DW_EH_PE_udata4 = 0x03,
  DW_EH_PE_udata8 = 0x04,
  DW_EH_PE_sleb128 = 0x09,
  DW_EH_PE_sdata2 = 0x0A,
  DW_EH_PE_sdata4 = 0x0B,
  DW_EH_PE_sdata8 = 0x0C,
  DW_EH_PE_signed = 0x08,
  DW_EH_PE_pcrel = 0x10,
  DW_EH_PE_textrel = 0x20,
  DW_EH_PE_datarel = 0x30,
  DW_EH_PE_funcrel = 0x40,
  DW_EH_PE_aligned = 0x50,
  DW_EH_PE_indirect = 0x80
};

enum ApplePropertyAttributes {
  // Apple Objective-C Property Attributes
  DW_APPLE_PROPERTY_readonly = 0x01,
  DW_APPLE_PROPERTY_readwrite = 0x02,
  DW_APPLE_PROPERTY_assign = 0x04,
  DW_APPLE_PROPERTY_retain = 0x08,
  DW_APPLE_PROPERTY_copy = 0x10,
  DW_APPLE_PROPERTY_nonatomic = 0x20
};

/// TagString - Return the string for the specified tag.
///
const char *TagString(unsigned Tag);

/// ChildrenString - Return the string for the specified children flag.
///
const char *ChildrenString(unsigned Children);

/// AttributeString - Return the string for the specified attribute.
///
const char *AttributeString(unsigned Attribute);

/// FormEncodingString - Return the string for the specified form encoding.
///
const char *FormEncodingString(unsigned Encoding);

/// OperationEncodingString - Return the string for the specified operation
/// encoding.
const char *OperationEncodingString(unsigned Encoding);

/// AttributeEncodingString - Return the string for the specified attribute
/// encoding.
const char *AttributeEncodingString(unsigned Encoding);

/// DecimalSignString - Return the string for the specified decimal sign
/// attribute.
const char *DecimalSignString(unsigned Sign);

/// EndianityString - Return the string for the specified endianity.
///
const char *EndianityString(unsigned Endian);

/// AccessibilityString - Return the string for the specified accessibility.
///
const char *AccessibilityString(unsigned Access);

/// VisibilityString - Return the string for the specified visibility.
///
const char *VisibilityString(unsigned Visibility);

/// VirtualityString - Return the string for the specified virtuality.
///
const char *VirtualityString(unsigned Virtuality);

/// LanguageString - Return the string for the specified language.
///
const char *LanguageString(unsigned Language);

/// CaseString - Return the string for the specified identifier case.
///
const char *CaseString(unsigned Case);

/// ConventionString - Return the string for the specified calling convention.
///
const char *ConventionString(unsigned Convention);

/// InlineCodeString - Return the string for the specified inline code.
///
const char *InlineCodeString(unsigned Code);

/// ArrayOrderString - Return the string for the specified array order.
///
const char *ArrayOrderString(unsigned Order);

/// DiscriminantString - Return the string for the specified discriminant
/// descriptor.
const char *DiscriminantString(unsigned Discriminant);

/// LNStandardString - Return the string for the specified line number standard.
///
const char *LNStandardString(unsigned Standard);

/// LNExtendedString - Return the string for the specified line number extended
/// opcode encodings.
const char *LNExtendedString(unsigned Encoding);

/// MacinfoString - Return the string for the specified macinfo type encodings.
///
const char *MacinfoString(unsigned Encoding);

/// CallFrameString - Return the string for the specified call frame instruction
/// encodings.
const char *CallFrameString(unsigned Encoding);

// Constants for the DWARF5 Accelerator Table Proposal
enum AcceleratorTable {
  // Data layout descriptors.
  DW_ATOM_null = 0u,       // Marker as the end of a list of atoms.
  DW_ATOM_die_offset = 1u, // DIE offset in the debug_info section.
  DW_ATOM_cu_offset = 2u, // Offset of the compile unit header that contains the
                          // item in question.
  DW_ATOM_die_tag = 3u,   // A tag entry.
  DW_ATOM_type_flags = 4u, // Set of flags for a type.

  // DW_ATOM_type_flags values.

  // Always set for C++, only set for ObjC if this is the @implementation for a
  // class.
  DW_FLAG_type_implementation = 2u,

  // Hash functions.

  // Daniel J. Bernstein hash.
  DW_hash_function_djb = 0u
};

/// AtomTypeString - Return the string for the specified Atom type.
const char *AtomTypeString(unsigned Atom);

// Constants for the GNU pubnames/pubtypes extensions supporting gdb index.
enum GDBIndexEntryKind {
  GIEK_NONE,
  GIEK_TYPE,
  GIEK_VARIABLE,
  GIEK_FUNCTION,
  GIEK_OTHER,
  GIEK_UNUSED5,
  GIEK_UNUSED6,
  GIEK_UNUSED7
};

const char *GDBIndexEntryKindString(GDBIndexEntryKind Kind);

enum GDBIndexEntryLinkage {
  GIEL_EXTERNAL,
  GIEL_STATIC
};

const char *GDBIndexEntryLinkageString(GDBIndexEntryLinkage Linkage);

/// The gnu_pub* kind looks like:
///
/// 0-3  reserved
/// 4-6  symbol kind
/// 7    0 == global, 1 == static
///
/// A gdb_index descriptor includes the above kind, shifted 24 bits up with the
/// offset of the cu within the debug_info section stored in those 24 bits.
struct PubIndexEntryDescriptor {
  GDBIndexEntryKind Kind;
  GDBIndexEntryLinkage Linkage;
  PubIndexEntryDescriptor(GDBIndexEntryKind Kind, GDBIndexEntryLinkage Linkage)
      : Kind(Kind), Linkage(Linkage) {}
  /* implicit */ PubIndexEntryDescriptor(GDBIndexEntryKind Kind)
      : Kind(Kind), Linkage(GIEL_EXTERNAL) {}
  explicit PubIndexEntryDescriptor(uint8_t Value)
      : Kind(static_cast<GDBIndexEntryKind>((Value & KIND_MASK) >>
                                            KIND_OFFSET)),
        Linkage(static_cast<GDBIndexEntryLinkage>((Value & LINKAGE_MASK) >>
                                                  LINKAGE_OFFSET)) {}
  uint8_t toBits() { return Kind << KIND_OFFSET | Linkage << LINKAGE_OFFSET; }

private:
  enum {
    KIND_OFFSET = 4,
    KIND_MASK = 7 << KIND_OFFSET,
    LINKAGE_OFFSET = 7,
    LINKAGE_MASK = 1 << LINKAGE_OFFSET
  };
};

} // End of namespace dwarf

} // End of namespace llvm

#endif
