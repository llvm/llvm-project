// RUN: llvm-mc -g -triple i386-apple-darwin10 %s -filetype=obj -o %t
// RUN: llvm-dwarfdump -all %t | FileCheck %s

.globl _bar
_bar:
	movl	$0, %eax
L1:	leave
	ret
_foo:
_baz:
	nop
.data
_x:	.long 1

// CHECK: file format Mach-O 32-bit i386

// CHECK: .debug_abbrev contents:
// CHECK: Abbrev table for offset: 0x00000000
// CHECK: [1] DW_TAG_compile_unit	DW_CHILDREN_yes
// CHECK: 	DW_AT_stmt_list	DW_FORM_sec_offset
// CHECK: 	DW_AT_low_pc	DW_FORM_addr
// CHECK: 	DW_AT_high_pc	DW_FORM_addr
// CHECK: 	DW_AT_name	DW_FORM_string
// CHECK: 	DW_AT_comp_dir	DW_FORM_string
// CHECK: 	DW_AT_producer	DW_FORM_string
// CHECK: 	DW_AT_language	DW_FORM_data2

// CHECK: [2] DW_TAG_label	DW_CHILDREN_yes
// CHECK: 	DW_AT_name	DW_FORM_string
// CHECK: 	DW_AT_decl_file	DW_FORM_data4
// CHECK: 	DW_AT_decl_line	DW_FORM_data4
// CHECK: 	DW_AT_low_pc	DW_FORM_addr
// CHECK: 	DW_AT_prototyped	DW_FORM_flag

// CHECK: [3] DW_TAG_unspecified_parameters	DW_CHILDREN_no


// CHECK: .debug_info contents:

// We don't check the leading addresses these are at.
// CHECK:  DW_TAG_compile_unit
// CHECK:    DW_AT_stmt_list (0x00000000)
// CHECK:    DW_AT_low_pc (0x0000000000000000)
// CHECK:    DW_AT_high_pc (0x0000000000000008)
// We don't check the file name as it is a temp directory
// CHECK:    DW_AT_name
// We don't check the DW_AT_comp_dir which is the current working directory
// CHECK:    DW_AT_producer ("llvm-mc (based on {{.*}})")
// CHECK:    DW_AT_language (DW_LANG_Mips_Assembler)

// CHECK:    DW_TAG_label
// CHECK:      DW_AT_name ("bar")
// CHECK:      DW_AT_decl_file ([[FILE:".*gen-dwarf.s"]])
// CHECK:      DW_AT_decl_line (5)
// CHECK:      DW_AT_low_pc (0x0000000000000000)
// CHECK:      DW_AT_prototyped (0x00)

// CHECK:      DW_TAG_unspecified_parameters

// CHECK:      NULL

// CHECK:    DW_TAG_label
// CHECK:      DW_AT_name ("foo")
// CHECK:      DW_AT_decl_file ([[FILE]])
// CHECK:      DW_AT_decl_line (9)
// CHECK:      DW_AT_low_pc (0x0000000000000007)
// CHECK:      DW_AT_prototyped (0x00)

// CHECK:      DW_TAG_unspecified_parameters

// CHECK:      NULL

// CHECK:    DW_TAG_label
// CHECK:      DW_AT_name ("baz")
// CHECK:      DW_AT_decl_file ([[FILE]])
// CHECK:      DW_AT_decl_line (10)
// CHECK:      DW_AT_low_pc (0x0000000000000007)
// CHECK:      DW_AT_prototyped (0x00)

// CHECK:      DW_TAG_unspecified_parameters

// CHECK:      NULL

// CHECK:    NULL

// CHECK: .debug_aranges contents:
// CHECK: Address Range Header: length = 0x0000001c, version = 0x0002, cu_offset = 0x00000000, addr_size = 0x04, seg_size = 0x00

// CHECK: .debug_line contents:
// CHECK: Line table prologue:
// We don't check the total_length as it includes lengths of temp paths
// CHECK:         version: 4
// We don't check the prologue_length as it too includes lengths of temp paths
// CHECK: min_inst_length: 1
// CHECK: default_is_stmt: 1
// CHECK:       line_base: -5
// CHECK:      line_range: 14
// CHECK:     opcode_base: 13
// CHECK: standard_opcode_lengths[DW_LNS_copy] = 0
// CHECK: standard_opcode_lengths[DW_LNS_advance_pc] = 1
// CHECK: standard_opcode_lengths[DW_LNS_advance_line] = 1
// CHECK: standard_opcode_lengths[DW_LNS_set_file] = 1
// CHECK: standard_opcode_lengths[DW_LNS_set_column] = 1
// CHECK: standard_opcode_lengths[DW_LNS_negate_stmt] = 0
// CHECK: standard_opcode_lengths[DW_LNS_set_basic_block] = 0
// CHECK: standard_opcode_lengths[DW_LNS_const_add_pc] = 0
// CHECK: standard_opcode_lengths[DW_LNS_fixed_advance_pc] = 1
// CHECK: standard_opcode_lengths[DW_LNS_set_prologue_end] = 0
// CHECK: standard_opcode_lengths[DW_LNS_set_epilogue_begin] = 0
// CHECK: standard_opcode_lengths[DW_LNS_set_isa] = 1
// We don't check include_directories as it has a temp path
// CHECK: file_names[  1]:
// CHECK-NEXT: name: "gen-dwarf.s"
// CHECK-NEXT: dir_index: 1

// CHECK: Address            Line   Column File   ISA Discriminator Flags
// CHECK: ------------------ ------ ------ ------ --- ------------- -------------
// CHECK: 0x0000000000000000      6      0      1   0             0  is_stmt
// CHECK: 0x0000000000000005      7      0      1   0             0  is_stmt
// CHECK: 0x0000000000000006      8      0      1   0             0  is_stmt
// CHECK: 0x0000000000000007     11      0      1   0             0  is_stmt
// CHECK: 0x0000000000000008     11      0      1   0             0  is_stmt end_sequence
