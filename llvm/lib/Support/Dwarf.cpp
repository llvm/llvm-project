//===-- llvm/Support/Dwarf.cpp - Dwarf Framework ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for generic dwarf information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Dwarf.h"

#include <cassert>

namespace llvm {

namespace dwarf {

/// TagString - Return the string for the specified tag.
///
const char *TagString(unsigned Tag) {
  switch(Tag) {
    case DW_TAG_array_type:                return "DW_TAG_array_type";
    case DW_TAG_class_type:                return "DW_TAG_class_type";
    case DW_TAG_entry_point:               return "DW_TAG_entry_point";
    case DW_TAG_enumeration_type:          return "DW_TAG_enumeration_type";
    case DW_TAG_formal_parameter:          return "DW_TAG_formal_parameter";
    case DW_TAG_imported_declaration:      return "DW_TAG_imported_declaration";
    case DW_TAG_label:                     return "DW_TAG_label";
    case DW_TAG_lexical_block:             return "DW_TAG_lexical_block";
    case DW_TAG_member:                    return "DW_TAG_member";
    case DW_TAG_pointer_type:              return "DW_TAG_pointer_type";
    case DW_TAG_reference_type:            return "DW_TAG_reference_type";
    case DW_TAG_compile_unit:              return "DW_TAG_compile_unit";
    case DW_TAG_string_type:               return "DW_TAG_string_type";
    case DW_TAG_structure_type:            return "DW_TAG_structure_type";
    case DW_TAG_subroutine_type:           return "DW_TAG_subroutine_type";
    case DW_TAG_typedef:                   return "DW_TAG_typedef";
    case DW_TAG_union_type:                return "DW_TAG_union_type";
    case DW_TAG_unspecified_parameters:    return "DW_TAG_unspecified_parameters";
    case DW_TAG_variant:                   return "DW_TAG_variant";
    case DW_TAG_common_block:              return "DW_TAG_common_block";
    case DW_TAG_common_inclusion:          return "DW_TAG_common_inclusion";
    case DW_TAG_inheritance:               return "DW_TAG_inheritance";
    case DW_TAG_inlined_subroutine:        return "DW_TAG_inlined_subroutine";
    case DW_TAG_module:                    return "DW_TAG_module";
    case DW_TAG_ptr_to_member_type:        return "DW_TAG_ptr_to_member_type";
    case DW_TAG_set_type:                  return "DW_TAG_set_type";
    case DW_TAG_subrange_type:             return "DW_TAG_subrange_type";
    case DW_TAG_with_stmt:                 return "DW_TAG_with_stmt";
    case DW_TAG_access_declaration:        return "DW_TAG_access_declaration";
    case DW_TAG_base_type:                 return "DW_TAG_base_type";
    case DW_TAG_catch_block:               return "DW_TAG_catch_block";
    case DW_TAG_const_type:                return "DW_TAG_const_type";
    case DW_TAG_constant:                  return "DW_TAG_constant";
    case DW_TAG_enumerator:                return "DW_TAG_enumerator";
    case DW_TAG_file_type:                 return "DW_TAG_file_type";
    case DW_TAG_friend:                    return "DW_TAG_friend";
    case DW_TAG_namelist:                  return "DW_TAG_namelist";
    case DW_TAG_namelist_item:             return "DW_TAG_namelist_item";
    case DW_TAG_packed_type:               return "DW_TAG_packed_type";
    case DW_TAG_subprogram:                return "DW_TAG_subprogram";
    case DW_TAG_template_type_parameter:   return "DW_TAG_template_type_parameter";
    case DW_TAG_template_value_parameter: return "DW_TAG_template_value_parameter";
    case DW_TAG_thrown_type:               return "DW_TAG_thrown_type";
    case DW_TAG_try_block:                 return "DW_TAG_try_block";
    case DW_TAG_variant_part:              return "DW_TAG_variant_part";
    case DW_TAG_variable:                  return "DW_TAG_variable";
    case DW_TAG_volatile_type:             return "DW_TAG_volatile_type";
    case DW_TAG_dwarf_procedure:           return "DW_TAG_dwarf_procedure";
    case DW_TAG_restrict_type:             return "DW_TAG_restrict_type";
    case DW_TAG_interface_type:            return "DW_TAG_interface_type";
    case DW_TAG_namespace:                 return "DW_TAG_namespace";
    case DW_TAG_imported_module:           return "DW_TAG_imported_module";
    case DW_TAG_unspecified_type:          return "DW_TAG_unspecified_type";
    case DW_TAG_partial_unit:              return "DW_TAG_partial_unit";
    case DW_TAG_imported_unit:             return "DW_TAG_imported_unit";
    case DW_TAG_condition:                 return "DW_TAG_condition";
    case DW_TAG_shared_type:               return "DW_TAG_shared_type";
    case DW_TAG_lo_user:                   return "DW_TAG_lo_user";
    case DW_TAG_hi_user:                   return "DW_TAG_hi_user";
  }
  assert(0 && "Unknown Dwarf Tag");
  return "";
}

/// ChildrenString - Return the string for the specified children flag.
///
const char *ChildrenString(unsigned Children) {
  switch(Children) {
    case DW_CHILDREN_no:                   return "CHILDREN_no";
    case DW_CHILDREN_yes:                  return "CHILDREN_yes";
  }
  assert(0 && "Unknown Dwarf ChildrenFlag");
  return "";
}

/// AttributeString - Return the string for the specified attribute.
///
const char *AttributeString(unsigned Attribute) {
  switch(Attribute) {
    case DW_AT_sibling:                    return "DW_AT_sibling";
    case DW_AT_location:                   return "DW_AT_location";
    case DW_AT_name:                       return "DW_AT_name";
    case DW_AT_ordering:                   return "DW_AT_ordering";
    case DW_AT_byte_size:                  return "DW_AT_byte_size";
    case DW_AT_bit_offset:                 return "DW_AT_bit_offset";
    case DW_AT_bit_size:                   return "DW_AT_bit_size";
    case DW_AT_stmt_list:                  return "DW_AT_stmt_list";
    case DW_AT_low_pc:                     return "DW_AT_low_pc";
    case DW_AT_high_pc:                    return "DW_AT_high_pc";
    case DW_AT_language:                   return "DW_AT_language";
    case DW_AT_discr:                      return "DW_AT_discr";
    case DW_AT_discr_value:                return "DW_AT_discr_value";
    case DW_AT_visibility:                 return "DW_AT_visibility";
    case DW_AT_import:                     return "DW_AT_import";
    case DW_AT_string_length:              return "DW_AT_string_length";
    case DW_AT_common_reference:           return "DW_AT_common_reference";
    case DW_AT_comp_dir:                   return "DW_AT_comp_dir";
    case DW_AT_const_value:                return "DW_AT_const_value";
    case DW_AT_containing_type:            return "DW_AT_containing_type";
    case DW_AT_default_value:              return "DW_AT_default_value";
    case DW_AT_inline:                     return "DW_AT_inline";
    case DW_AT_is_optional:                return "DW_AT_is_optional";
    case DW_AT_lower_bound:                return "DW_AT_lower_bound";
    case DW_AT_producer:                   return "DW_AT_producer";
    case DW_AT_prototyped:                 return "DW_AT_prototyped";
    case DW_AT_return_addr:                return "DW_AT_return_addr";
    case DW_AT_start_scope:                return "DW_AT_start_scope";
    case DW_AT_bit_stride:                 return "DW_AT_bit_stride";
    case DW_AT_upper_bound:                return "DW_AT_upper_bound";
    case DW_AT_abstract_origin:            return "DW_AT_abstract_origin";
    case DW_AT_accessibility:              return "DW_AT_accessibility";
    case DW_AT_address_class:              return "DW_AT_address_class";
    case DW_AT_artificial:                 return "DW_AT_artificial";
    case DW_AT_base_types:                 return "DW_AT_base_types";
    case DW_AT_calling_convention:         return "DW_AT_calling_convention";
    case DW_AT_count:                      return "DW_AT_count";
    case DW_AT_data_member_location:       return "DW_AT_data_member_location";
    case DW_AT_decl_column:                return "DW_AT_decl_column";
    case DW_AT_decl_file:                  return "DW_AT_decl_file";
    case DW_AT_decl_line:                  return "DW_AT_decl_line";
    case DW_AT_declaration:                return "DW_AT_declaration";
    case DW_AT_discr_list:                 return "DW_AT_discr_list";
    case DW_AT_encoding:                   return "DW_AT_encoding";
    case DW_AT_external:                   return "DW_AT_external";
    case DW_AT_frame_base:                 return "DW_AT_frame_base";
    case DW_AT_friend:                     return "DW_AT_friend";
    case DW_AT_identifier_case:            return "DW_AT_identifier_case";
    case DW_AT_macro_info:                 return "DW_AT_macro_info";
    case DW_AT_namelist_item:              return "DW_AT_namelist_item";
    case DW_AT_priority:                   return "DW_AT_priority";
    case DW_AT_segment:                    return "DW_AT_segment";
    case DW_AT_specification:              return "DW_AT_specification";
    case DW_AT_static_link:                return "DW_AT_static_link";
    case DW_AT_type:                       return "DW_AT_type";
    case DW_AT_use_location:               return "DW_AT_use_location";
    case DW_AT_variable_parameter:         return "DW_AT_variable_parameter";
    case DW_AT_virtuality:                 return "DW_AT_virtuality";
    case DW_AT_vtable_elem_location:       return "DW_AT_vtable_elem_location";
    case DW_AT_allocated:                  return "DW_AT_allocated";
    case DW_AT_associated:                 return "DW_AT_associated";
    case DW_AT_data_location:              return "DW_AT_data_location";
    case DW_AT_byte_stride:                return "DW_AT_byte_stride";
    case DW_AT_entry_pc:                   return "DW_AT_entry_pc";
    case DW_AT_use_UTF8:                   return "DW_AT_use_UTF8";
    case DW_AT_extension:                  return "DW_AT_extension";
    case DW_AT_ranges:                     return "DW_AT_ranges";
    case DW_AT_trampoline:                 return "DW_AT_trampoline";
    case DW_AT_call_column:                return "DW_AT_call_column";
    case DW_AT_call_file:                  return "DW_AT_call_file";
    case DW_AT_call_line:                  return "DW_AT_call_line";
    case DW_AT_description:                return "DW_AT_description";
    case DW_AT_binary_scale:               return "DW_AT_binary_scale";
    case DW_AT_decimal_scale:              return "DW_AT_decimal_scale";
    case DW_AT_small:                      return "DW_AT_small";
    case DW_AT_decimal_sign:               return "DW_AT_decimal_sign";
    case DW_AT_digit_count:                return "DW_AT_digit_count";
    case DW_AT_picture_string:             return "DW_AT_picture_string";
    case DW_AT_mutable:                    return "DW_AT_mutable";
    case DW_AT_threads_scaled:             return "DW_AT_threads_scaled";
    case DW_AT_explicit:                   return "DW_AT_explicit";
    case DW_AT_object_pointer:             return "DW_AT_object_pointer";
    case DW_AT_endianity:                  return "DW_AT_endianity";
    case DW_AT_elemental:                  return "DW_AT_elemental";
    case DW_AT_pure:                       return "DW_AT_pure";
    case DW_AT_recursive:                  return "DW_AT_recursive";
    case DW_AT_MIPS_linkage_name:          return "DW_AT_MIPS_linkage_name";
    case DW_AT_sf_names:                   return "DW_AT_sf_names";
    case DW_AT_src_info:                   return "DW_AT_src_info";
    case DW_AT_mac_info:                   return "DW_AT_mac_info";
    case DW_AT_src_coords:                 return "DW_AT_src_coords";
    case DW_AT_body_begin:                 return "DW_AT_body_begin";
    case DW_AT_body_end:                   return "DW_AT_body_end";
    case DW_AT_GNU_vector:                 return "DW_AT_GNU_vector";
    case DW_AT_lo_user:                    return "DW_AT_lo_user";
    case DW_AT_hi_user:                    return "DW_AT_hi_user";
    case DW_AT_APPLE_optimized:            return "DW_AT_APPLE_optimized";
    case DW_AT_APPLE_flags:                return "DW_AT_APPLE_flags";
  }
  assert(0 && "Unknown Dwarf Attribute");
  return "";
}

/// FormEncodingString - Return the string for the specified form encoding.
///
const char *FormEncodingString(unsigned Encoding) {
  switch(Encoding) {
    case DW_FORM_addr:                     return "FORM_addr";
    case DW_FORM_block2:                   return "FORM_block2";
    case DW_FORM_block4:                   return "FORM_block4";
    case DW_FORM_data2:                    return "FORM_data2";
    case DW_FORM_data4:                    return "FORM_data4";
    case DW_FORM_data8:                    return "FORM_data8";
    case DW_FORM_string:                   return "FORM_string";
    case DW_FORM_block:                    return "FORM_block";
    case DW_FORM_block1:                   return "FORM_block1";
    case DW_FORM_data1:                    return "FORM_data1";
    case DW_FORM_flag:                     return "FORM_flag";
    case DW_FORM_sdata:                    return "FORM_sdata";
    case DW_FORM_strp:                     return "FORM_strp";
    case DW_FORM_udata:                    return "FORM_udata";
    case DW_FORM_ref_addr:                 return "FORM_ref_addr";
    case DW_FORM_ref1:                     return "FORM_ref1";
    case DW_FORM_ref2:                     return "FORM_ref2";
    case DW_FORM_ref4:                     return "FORM_ref4";
    case DW_FORM_ref8:                     return "FORM_ref8";
    case DW_FORM_ref_udata:                return "FORM_ref_udata";
    case DW_FORM_indirect:                 return "FORM_indirect";
  }
  assert(0 && "Unknown Dwarf Form Encoding");
  return "";
}

/// OperationEncodingString - Return the string for the specified operation
/// encoding.
const char *OperationEncodingString(unsigned Encoding) {
  switch(Encoding) {
    case DW_OP_addr:                       return "OP_addr";
    case DW_OP_deref:                      return "OP_deref";
    case DW_OP_const1u:                    return "OP_const1u";
    case DW_OP_const1s:                    return "OP_const1s";
    case DW_OP_const2u:                    return "OP_const2u";
    case DW_OP_const2s:                    return "OP_const2s";
    case DW_OP_const4u:                    return "OP_const4u";
    case DW_OP_const4s:                    return "OP_const4s";
    case DW_OP_const8u:                    return "OP_const8u";
    case DW_OP_const8s:                    return "OP_const8s";
    case DW_OP_constu:                     return "OP_constu";
    case DW_OP_consts:                     return "OP_consts";
    case DW_OP_dup:                        return "OP_dup";
    case DW_OP_drop:                       return "OP_drop";
    case DW_OP_over:                       return "OP_over";
    case DW_OP_pick:                       return "OP_pick";
    case DW_OP_swap:                       return "OP_swap";
    case DW_OP_rot:                        return "OP_rot";
    case DW_OP_xderef:                     return "OP_xderef";
    case DW_OP_abs:                        return "OP_abs";
    case DW_OP_and:                        return "OP_and";
    case DW_OP_div:                        return "OP_div";
    case DW_OP_minus:                      return "OP_minus";
    case DW_OP_mod:                        return "OP_mod";
    case DW_OP_mul:                        return "OP_mul";
    case DW_OP_neg:                        return "OP_neg";
    case DW_OP_not:                        return "OP_not";
    case DW_OP_or:                         return "OP_or";
    case DW_OP_plus:                       return "OP_plus";
    case DW_OP_plus_uconst:                return "OP_plus_uconst";
    case DW_OP_shl:                        return "OP_shl";
    case DW_OP_shr:                        return "OP_shr";
    case DW_OP_shra:                       return "OP_shra";
    case DW_OP_xor:                        return "OP_xor";
    case DW_OP_skip:                       return "OP_skip";
    case DW_OP_bra:                        return "OP_bra";
    case DW_OP_eq:                         return "OP_eq";
    case DW_OP_ge:                         return "OP_ge";
    case DW_OP_gt:                         return "OP_gt";
    case DW_OP_le:                         return "OP_le";
    case DW_OP_lt:                         return "OP_lt";
    case DW_OP_ne:                         return "OP_ne";
    case DW_OP_lit0:                       return "OP_lit0";
    case DW_OP_lit1:                       return "OP_lit1";
    case DW_OP_lit31:                      return "OP_lit31";
    case DW_OP_reg0:                       return "OP_reg0";
    case DW_OP_reg1:                       return "OP_reg1";
    case DW_OP_reg31:                      return "OP_reg31";
    case DW_OP_breg0:                      return "OP_breg0";
    case DW_OP_breg1:                      return "OP_breg1";
    case DW_OP_breg31:                     return "OP_breg31";
    case DW_OP_regx:                       return "OP_regx";
    case DW_OP_fbreg:                      return "OP_fbreg";
    case DW_OP_bregx:                      return "OP_bregx";
    case DW_OP_piece:                      return "OP_piece";
    case DW_OP_deref_size:                 return "OP_deref_size";
    case DW_OP_xderef_size:                return "OP_xderef_size";
    case DW_OP_nop:                        return "OP_nop";
    case DW_OP_push_object_address:        return "OP_push_object_address";
    case DW_OP_call2:                      return "OP_call2";
    case DW_OP_call4:                      return "OP_call4";
    case DW_OP_call_ref:                   return "OP_call_ref";
    case DW_OP_form_tls_address:           return "OP_form_tls_address";
    case DW_OP_call_frame_cfa:             return "OP_call_frame_cfa";
    case DW_OP_lo_user:                    return "OP_lo_user";
    case DW_OP_hi_user:                    return "OP_hi_user";
  }
  assert(0 && "Unknown Dwarf Operation Encoding");
  return "";
}

/// AttributeEncodingString - Return the string for the specified attribute
/// encoding.
const char *AttributeEncodingString(unsigned Encoding) {
  switch(Encoding) {
    case DW_ATE_address:                   return "ATE_address";
    case DW_ATE_boolean:                   return "ATE_boolean";
    case DW_ATE_complex_float:             return "ATE_complex_float";
    case DW_ATE_float:                     return "ATE_float";
    case DW_ATE_signed:                    return "ATE_signed";
    case DW_ATE_signed_char:               return "ATE_signed_char";
    case DW_ATE_unsigned:                  return "ATE_unsigned";
    case DW_ATE_unsigned_char:             return "ATE_unsigned_char";
    case DW_ATE_imaginary_float:           return "ATE_imaginary_float";
    case DW_ATE_packed_decimal:            return "ATE_packed_decimal";
    case DW_ATE_numeric_string:            return "ATE_numeric_string";
    case DW_ATE_edited:                    return "ATE_edited";
    case DW_ATE_signed_fixed:              return "ATE_signed_fixed";
    case DW_ATE_unsigned_fixed:            return "ATE_unsigned_fixed";
    case DW_ATE_decimal_float:             return "ATE_decimal_float";
    case DW_ATE_lo_user:                   return "ATE_lo_user";
    case DW_ATE_hi_user:                   return "ATE_hi_user";
  }
  assert(0 && "Unknown Dwarf Attribute Encoding");
  return "";
}

/// DecimalSignString - Return the string for the specified decimal sign
/// attribute.
const char *DecimalSignString(unsigned Sign) {
  switch(Sign) {
    case DW_DS_unsigned:                   return "DS_unsigned";
    case DW_DS_leading_overpunch:          return "DS_leading_overpunch";
    case DW_DS_trailing_overpunch:         return "DS_trailing_overpunch";
    case DW_DS_leading_separate:           return "DS_leading_separate";
    case DW_DS_trailing_separate:          return "DS_trailing_separate";
  }
  assert(0 && "Unknown Dwarf Decimal Sign Attribute");
  return "";
}

/// EndianityString - Return the string for the specified endianity.
///
const char *EndianityString(unsigned Endian) {
  switch(Endian) {
    case DW_END_default:                   return "END_default";
    case DW_END_big:                       return "END_big";
    case DW_END_little:                    return "END_little";
    case DW_END_lo_user:                   return "END_lo_user";
    case DW_END_hi_user:                   return "END_hi_user";
  }
  assert(0 && "Unknown Dwarf Endianity");
  return "";
}

/// AccessibilityString - Return the string for the specified accessibility.
///
const char *AccessibilityString(unsigned Access) {
  switch(Access) {
    // Accessibility codes
    case DW_ACCESS_public:                 return "ACCESS_public";
    case DW_ACCESS_protected:              return "ACCESS_protected";
    case DW_ACCESS_private:                return "ACCESS_private";
  }
  assert(0 && "Unknown Dwarf Accessibility");
  return "";
}

/// VisibilityString - Return the string for the specified visibility.
///
const char *VisibilityString(unsigned Visibility) {
  switch(Visibility) {
    case DW_VIS_local:                     return "VIS_local";
    case DW_VIS_exported:                  return "VIS_exported";
    case DW_VIS_qualified:                 return "VIS_qualified";
  }
  assert(0 && "Unknown Dwarf Visibility");
  return "";
}

/// VirtualityString - Return the string for the specified virtuality.
///
const char *VirtualityString(unsigned Virtuality) {
  switch(Virtuality) {
    case DW_VIRTUALITY_none:               return "VIRTUALITY_none";
    case DW_VIRTUALITY_virtual:            return "VIRTUALITY_virtual";
    case DW_VIRTUALITY_pure_virtual:       return "VIRTUALITY_pure_virtual";
  }
  assert(0 && "Unknown Dwarf Virtuality");
  return "";
}

/// LanguageString - Return the string for the specified language.
///
const char *LanguageString(unsigned Language) {
  switch(Language) {
    case DW_LANG_C89:                      return "LANG_C89";
    case DW_LANG_C:                        return "LANG_C";
    case DW_LANG_Ada83:                    return "LANG_Ada83";
    case DW_LANG_C_plus_plus:              return "LANG_C_plus_plus";
    case DW_LANG_Cobol74:                  return "LANG_Cobol74";
    case DW_LANG_Cobol85:                  return "LANG_Cobol85";
    case DW_LANG_Fortran77:                return "LANG_Fortran77";
    case DW_LANG_Fortran90:                return "LANG_Fortran90";
    case DW_LANG_Pascal83:                 return "LANG_Pascal83";
    case DW_LANG_Modula2:                  return "LANG_Modula2";
    case DW_LANG_Java:                     return "LANG_Java";
    case DW_LANG_C99:                      return "LANG_C99";
    case DW_LANG_Ada95:                    return "LANG_Ada95";
    case DW_LANG_Fortran95:                return "LANG_Fortran95";
    case DW_LANG_PLI:                      return "LANG_PLI";
    case DW_LANG_ObjC:                     return "LANG_ObjC";
    case DW_LANG_ObjC_plus_plus:           return "LANG_ObjC_plus_plus";
    case DW_LANG_UPC:                      return "LANG_UPC";
    case DW_LANG_D:                        return "LANG_D";
    case DW_LANG_lo_user:                  return "LANG_lo_user";
    case DW_LANG_hi_user:                  return "LANG_hi_user";
  }
  assert(0 && "Unknown Dwarf Language");
  return "";
}

/// CaseString - Return the string for the specified identifier case.
///
const char *CaseString(unsigned Case) {
   switch(Case) {
    case DW_ID_case_sensitive:             return "ID_case_sensitive";
    case DW_ID_up_case:                    return "ID_up_case";
    case DW_ID_down_case:                  return "ID_down_case";
    case DW_ID_case_insensitive:           return "ID_case_insensitive";
  }
  assert(0 && "Unknown Dwarf Identifier Case");
  return "";
}

/// ConventionString - Return the string for the specified calling convention.
///
const char *ConventionString(unsigned Convention) {
   switch(Convention) {
    case DW_CC_normal:                     return "CC_normal";
    case DW_CC_program:                    return "CC_program";
    case DW_CC_nocall:                     return "CC_nocall";
    case DW_CC_lo_user:                    return "CC_lo_user";
    case DW_CC_hi_user:                    return "CC_hi_user";
  }
  assert(0 && "Unknown Dwarf Calling Convention");
  return "";
}

/// InlineCodeString - Return the string for the specified inline code.
///
const char *InlineCodeString(unsigned Code) {
   switch(Code) {
    case DW_INL_not_inlined:               return "INL_not_inlined";
    case DW_INL_inlined:                   return "INL_inlined";
    case DW_INL_declared_not_inlined:      return "INL_declared_not_inlined";
    case DW_INL_declared_inlined:          return "INL_declared_inlined";
  }
  assert(0 && "Unknown Dwarf Inline Code");
  return "";
}

/// ArrayOrderString - Return the string for the specified array order.
///
const char *ArrayOrderString(unsigned Order) {
   switch(Order) {
    case DW_ORD_row_major:                 return "ORD_row_major";
    case DW_ORD_col_major:                 return "ORD_col_major";
  }
  assert(0 && "Unknown Dwarf Array Order");
  return "";
}

/// DiscriminantString - Return the string for the specified discriminant
/// descriptor.
const char *DiscriminantString(unsigned Discriminant) {
   switch(Discriminant) {
    case DW_DSC_label:                     return "DSC_label";
    case DW_DSC_range:                     return "DSC_range";
  }
  assert(0 && "Unknown Dwarf Discriminant Descriptor");
  return "";
}

/// LNStandardString - Return the string for the specified line number standard.
///
const char *LNStandardString(unsigned Standard) {
   switch(Standard) {
    case DW_LNS_copy:                      return "LNS_copy";
    case DW_LNS_advance_pc:                return "LNS_advance_pc";
    case DW_LNS_advance_line:              return "LNS_advance_line";
    case DW_LNS_set_file:                  return "LNS_set_file";
    case DW_LNS_set_column:                return "LNS_set_column";
    case DW_LNS_negate_stmt:               return "LNS_negate_stmt";
    case DW_LNS_set_basic_block:           return "LNS_set_basic_block";
    case DW_LNS_const_add_pc:              return "LNS_const_add_pc";
    case DW_LNS_fixed_advance_pc:          return "LNS_fixed_advance_pc";
    case DW_LNS_set_prologue_end:          return "LNS_set_prologue_end";
    case DW_LNS_set_epilogue_begin:        return "LNS_set_epilogue_begin";
    case DW_LNS_set_isa:                   return "LNS_set_isa";
  }
  assert(0 && "Unknown Dwarf Line Number Standard");
  return "";
}

/// LNExtendedString - Return the string for the specified line number extended
/// opcode encodings.
const char *LNExtendedString(unsigned Encoding) {
   switch(Encoding) {
    // Line Number Extended Opcode Encodings
    case DW_LNE_end_sequence:              return "LNE_end_sequence";
    case DW_LNE_set_address:               return "LNE_set_address";
    case DW_LNE_define_file:               return "LNE_define_file";
    case DW_LNE_lo_user:                   return "LNE_lo_user";
    case DW_LNE_hi_user:                   return "LNE_hi_user";
  }
  assert(0 && "Unknown Dwarf Line Number Extended Opcode Encoding");
  return "";
}

/// MacinfoString - Return the string for the specified macinfo type encodings.
///
const char *MacinfoString(unsigned Encoding) {
   switch(Encoding) {
    // Macinfo Type Encodings
    case DW_MACINFO_define:                return "MACINFO_define";
    case DW_MACINFO_undef:                 return "MACINFO_undef";
    case DW_MACINFO_start_file:            return "MACINFO_start_file";
    case DW_MACINFO_end_file:              return "MACINFO_end_file";
    case DW_MACINFO_vendor_ext:            return "MACINFO_vendor_ext";
  }
  assert(0 && "Unknown Dwarf Macinfo Type Encodings");
  return "";
}

/// CallFrameString - Return the string for the specified call frame instruction
/// encodings.
const char *CallFrameString(unsigned Encoding) {
   switch(Encoding) {
    case DW_CFA_advance_loc:               return "CFA_advance_loc";
    case DW_CFA_offset:                    return "CFA_offset";
    case DW_CFA_restore:                   return "CFA_restore";
    case DW_CFA_set_loc:                   return "CFA_set_loc";
    case DW_CFA_advance_loc1:              return "CFA_advance_loc1";
    case DW_CFA_advance_loc2:              return "CFA_advance_loc2";
    case DW_CFA_advance_loc4:              return "CFA_advance_loc4";
    case DW_CFA_offset_extended:           return "CFA_offset_extended";
    case DW_CFA_restore_extended:          return "CFA_restore_extended";
    case DW_CFA_undefined:                 return "CFA_undefined";
    case DW_CFA_same_value:                return "CFA_same_value";
    case DW_CFA_register:                  return "CFA_register";
    case DW_CFA_remember_state:            return "CFA_remember_state";
    case DW_CFA_restore_state:             return "CFA_restore_state";
    case DW_CFA_def_cfa:                   return "CFA_def_cfa";
    case DW_CFA_def_cfa_register:          return "CFA_def_cfa_register";
    case DW_CFA_def_cfa_offset:            return "CFA_def_cfa_offset";
    case DW_CFA_def_cfa_expression:        return "CFA_def_cfa_expression";
    case DW_CFA_expression:                return "CFA_expression";
    case DW_CFA_offset_extended_sf:        return "CFA_offset_extended_sf";
    case DW_CFA_def_cfa_sf:                return "CFA_def_cfa_sf";
    case DW_CFA_def_cfa_offset_sf:         return "CFA_def_cfa_offset_sf";
    case DW_CFA_val_offset:                return "CFA_val_offset";
    case DW_CFA_val_offset_sf:             return "CFA_val_offset_sf";
    case DW_CFA_val_expression:            return "CFA_val_expression";
    case DW_CFA_lo_user:                   return "CFA_lo_user";
    case DW_CFA_hi_user:                   return "CFA_hi_user";
  }
  assert(0 && "Unknown Dwarf Call Frame Instruction Encodings");
  return "";
}

} // End of namespace dwarf.

} // End of namespace llvm.
