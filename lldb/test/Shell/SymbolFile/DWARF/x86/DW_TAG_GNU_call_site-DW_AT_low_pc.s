# This tests that lldb is compatible with DWARF-4 entry values GNU extension
# with DW_TAG_GNU_call_site attributes order as produced by GCC:
# 0x000000b1:     DW_TAG_GNU_call_site
#                   DW_AT_low_pc  (0x000000000040111e)
#                   DW_AT_abstract_origin (0x000000cc "a")
# clang produces the attributes in opposite order:
# 0x00000064:     DW_TAG_GNU_call_site
#                   DW_AT_abstract_origin (0x0000002a "a")
#                   DW_AT_low_pc  (0x0000000000401146)

# REQUIRES: target-x86_64, system-linux, lld

# RUN: %clang_host -o %t %s
# RUN: %lldb %t -o r -o 'p p' -o exit | FileCheck %s

# CHECK: (int) $0 = 1

# The DWARF has been produced and modified from:
# static __attribute__((noinline, noclone)) void b(int x) {
#   asm("");
# }
# static __attribute__((noinline, noclone)) void a(int p) {
#   b(2);
# }
# int main() {
#   a(1);
#   return 0;
# }

	.text
.Ltext0:
	.type	b, @function
b:
	.cfi_startproc
	ret
	.cfi_endproc
	.size	b, .-b
	.type	a, @function
a:
.LVL1:
.LFB1:
	.cfi_startproc
	movl	$2, %edi
.LVL2:
	call	b
	int3
	ret
	.cfi_endproc
.LFE1:
	.size	a, .-a
	.globl	main
	.type	main, @function
main:
.LFB2:
	.cfi_startproc
	movl	$1, %edi
	call	a
.LVL4:
	movl	$0, %eax
	ret
	.cfi_endproc
.LFE2:
	.size	main, .-main
.Letext0:
	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.long	.Ldebuginfo_end - .Ldebuginfo_start	# Length of Compilation Unit Info
.Ldebuginfo_start:
	.value	0x4	# DWARF version number
	.long	.Ldebug_abbrev0	# Offset Into Abbrev. Section
	.byte	0x8	# Pointer Size (in bytes)
	.uleb128 0x1	# (DIE (0xb) DW_TAG_compile_unit)
	.asciz "GNU C17 10.1.1 20200507 (Red Hat 10.1.1-1) -mtune=generic -march=x86-64 -g -Og"	# DW_AT_producer: "GNU C17 10.1.1 20200507 (Red Hat 10.1.1-1) -mtune=generic -march=x86-64 -g -Og"
	.byte	0xc	# DW_AT_language
	.asciz "DW_TAG_GNU_call_site-DW_AT_low_pc.c"	# DW_AT_name
	.asciz ""	# DW_AT_comp_dir: "/home/jkratoch/t"
	.quad	.Ltext0	# DW_AT_low_pc
	.quad	.Letext0-.Ltext0	# DW_AT_high_pc
	.uleb128 0x2	# (DIE (0x2d) DW_TAG_subprogram)
			# DW_AT_external
	.asciz "main"	# DW_AT_name: "main"
	.long	.Ltype_int - .Ldebug_info0	# DW_AT_type
	.quad	.LFB2	# DW_AT_low_pc
	.quad	.LFE2-.LFB2	# DW_AT_high_pc
	.uleb128 0x1	# DW_AT_frame_base
	.byte	0x9c	# DW_OP_call_frame_cfa
			# DW_AT_GNU_all_call_sites
	.uleb128 0x3	# (DIE (0x4f) DW_TAG_GNU_call_site)
	.quad	.LVL4	# DW_AT_low_pc
	.long	.Lfunc_a - .Ldebug_info0	# DW_AT_abstract_origin
	.uleb128 0x4	# (DIE (0x5c) DW_TAG_GNU_call_site_parameter)
	.uleb128 0x1	# DW_AT_location
	.byte	0x55	# DW_OP_reg5
	.uleb128 0x1	# DW_AT_GNU_call_site_value
	.byte	0x31	# DW_OP_lit1
	.byte	0	# end of children of DIE 0x4f
	.byte	0	# end of children of DIE 0x2d
.Ltype_int:
	.uleb128 0x5	# (DIE (0x63) DW_TAG_base_type)
	.byte	0x4	# DW_AT_byte_size
	.byte	0x5	# DW_AT_encoding
	.asciz "int"	# DW_AT_name
.Lfunc_a:
	.uleb128 0x6	# (DIE (0x6a) DW_TAG_subprogram)
	.asciz "a"	# DW_AT_name
			# DW_AT_prototyped
	.quad	.LFB1	# DW_AT_low_pc
	.quad	.LFE1-.LFB1	# DW_AT_high_pc
	.uleb128 0x1	# DW_AT_frame_base
	.byte	0x9c	# DW_OP_call_frame_cfa
			# DW_AT_GNU_all_call_sites
	.uleb128 0x7	# (DIE (0x86) DW_TAG_formal_parameter)
	.asciz "p"	# DW_AT_name
	.long	.Ltype_int - .Ldebug_info0	# DW_AT_type
	.long	.LLST0	# DW_AT_location
	.byte	0	# end of children of DIE 0x6a
	.byte	0	# end of children of DIE 0xb
.Ldebuginfo_end:
	.section	.debug_abbrev,"",@progbits
.Ldebug_abbrev0:
	.uleb128 0x1	# (abbrev code)
	.uleb128 0x11	# (TAG: DW_TAG_compile_unit)
	.byte	0x1	# DW_children_yes
	.uleb128 0x25	# (DW_AT_producer)
	.uleb128 0x8	# (DW_FORM_string)
	.uleb128 0x13	# (DW_AT_language)
	.uleb128 0xb	# (DW_FORM_data1)
	.uleb128 0x3	# (DW_AT_name)
	.uleb128 0x8	# (DW_FORM_string)
	.uleb128 0x1b	# (DW_AT_comp_dir)
	.uleb128 0x8	# (DW_FORM_string)
	.uleb128 0x11	# (DW_AT_low_pc)
	.uleb128 0x1	# (DW_FORM_addr)
	.uleb128 0x12	# (DW_AT_high_pc)
	.uleb128 0x7	# (DW_FORM_data8)
	.byte	0
	.byte	0
	.uleb128 0x2	# (abbrev code)
	.uleb128 0x2e	# (TAG: DW_TAG_subprogram)
	.byte	0x1	# DW_children_yes
	.uleb128 0x3f	# (DW_AT_external)
	.uleb128 0x19	# (DW_FORM_flag_present)
	.uleb128 0x3	# (DW_AT_name)
	.uleb128 0x8	# (DW_FORM_string)
	.uleb128 0x49	# (DW_AT_type)
	.uleb128 0x13	# (DW_FORM_ref4)
	.uleb128 0x11	# (DW_AT_low_pc)
	.uleb128 0x1	# (DW_FORM_addr)
	.uleb128 0x12	# (DW_AT_high_pc)
	.uleb128 0x7	# (DW_FORM_data8)
	.uleb128 0x40	# (DW_AT_frame_base)
	.uleb128 0x18	# (DW_FORM_exprloc)
	.uleb128 0x2117	# (DW_AT_GNU_all_call_sites)
	.uleb128 0x19	# (DW_FORM_flag_present)
	.byte	0
	.byte	0
	.uleb128 0x3	# (abbrev code)
	.uleb128 0x4109	# (TAG: DW_TAG_GNU_call_site)
	.byte	0x1	# DW_children_yes
	.uleb128 0x11	# (DW_AT_low_pc)
	.uleb128 0x1	# (DW_FORM_addr)
	.uleb128 0x31	# (DW_AT_abstract_origin)
	.uleb128 0x13	# (DW_FORM_ref4)
	.byte	0
	.byte	0
	.uleb128 0x4	# (abbrev code)
	.uleb128 0x410a	# (TAG: DW_TAG_GNU_call_site_parameter)
	.byte	0	# DW_children_no
	.uleb128 0x2	# (DW_AT_location)
	.uleb128 0x18	# (DW_FORM_exprloc)
	.uleb128 0x2111	# (DW_AT_GNU_call_site_value)
	.uleb128 0x18	# (DW_FORM_exprloc)
	.byte	0
	.byte	0
	.uleb128 0x5	# (abbrev code)
	.uleb128 0x24	# (TAG: DW_TAG_base_type)
	.byte	0	# DW_children_no
	.uleb128 0xb	# (DW_AT_byte_size)
	.uleb128 0xb	# (DW_FORM_data1)
	.uleb128 0x3e	# (DW_AT_encoding)
	.uleb128 0xb	# (DW_FORM_data1)
	.uleb128 0x3	# (DW_AT_name)
	.uleb128 0x8	# (DW_FORM_string)
	.byte	0
	.byte	0
	.uleb128 0x6	# (abbrev code)
	.uleb128 0x2e	# (TAG: DW_TAG_subprogram)
	.byte	0x1	# DW_children_yes
	.uleb128 0x3	# (DW_AT_name)
	.uleb128 0x8	# (DW_FORM_string)
	.uleb128 0x27	# (DW_AT_prototyped)
	.uleb128 0x19	# (DW_FORM_flag_present)
	.uleb128 0x11	# (DW_AT_low_pc)
	.uleb128 0x1	# (DW_FORM_addr)
	.uleb128 0x12	# (DW_AT_high_pc)
	.uleb128 0x7	# (DW_FORM_data8)
	.uleb128 0x40	# (DW_AT_frame_base)
	.uleb128 0x18	# (DW_FORM_exprloc)
	.uleb128 0x2117	# (DW_AT_GNU_all_call_sites)
	.uleb128 0x19	# (DW_FORM_flag_present)
	.byte	0
	.byte	0
	.uleb128 0x7	# (abbrev code)
	.uleb128 0x5	# (TAG: DW_TAG_formal_parameter)
	.byte	0	# DW_children_no
	.uleb128 0x3	# (DW_AT_name)
	.uleb128 0x8	# (DW_FORM_string)
	.uleb128 0x49	# (DW_AT_type)
	.uleb128 0x13	# (DW_FORM_ref4)
	.uleb128 0x2	# (DW_AT_location)
	.uleb128 0x17	# (DW_FORM_sec_offset)
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_loc,"",@progbits
.LLST0:
	.quad	.LVL1-.Ltext0	# Location list begin address (*.LLST0)
	.quad	.LVL2-.Ltext0	# Location list end address (*.LLST0)
	.value	0x1	# Location expression size
	.byte	0x55	# DW_OP_reg5
	.quad	.LVL2-.Ltext0	# Location list begin address (*.LLST0)
	.quad	.LFE1-.Ltext0	# Location list end address (*.LLST0)
	.value	0x4	# Location expression size
	.byte	0xf3	# DW_OP_GNU_entry_value
	.uleb128 0x1
	.byte	0x55	# DW_OP_reg5
	.byte	0x9f	# DW_OP_stack_value
	.quad	0	# Location list terminator begin (*.LLST0)
	.quad	0	# Location list terminator end (*.LLST0)
	.section	.note.GNU-stack,"",@progbits
