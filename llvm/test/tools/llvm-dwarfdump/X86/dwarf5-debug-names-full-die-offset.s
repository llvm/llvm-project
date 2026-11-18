## Test that full DIE offset is printed out for CU and local TU in the comment form.

# RUN: llvm-mc %s -filetype obj -triple x86_64-unknown-linux-gnu -o %t.o
# RUN: ld.lld %t.o -o %t.exe
# RUN: llvm-dwarfdump -debug-info -debug-names %t.exe | FileCheck %s

# CHECK: Type Unit:
# CHECK: [[OFFSET:0x[0-9a-f]*]]:   DW_TAG_class_type
# CHECK: [[OFFSET1:0x[0-9a-f]*]]:   DW_TAG_base_type
# CHECK: Compile Unit
# CHECK: [[OFFSET2:0x[0-9a-f]*]]:   DW_TAG_variable
# CHECK: [[OFFSET3:0x[0-9a-f]*]]:   DW_TAG_base_type
# CHECK: [[OFFSET4:0x[0-9a-f]*]]:   DW_TAG_variable

# CHECK: Bucket 0
# CHECK: Tag: DW_TAG_variable
# CHECK-NEXT: DW_IDX_die_offset: 0x0000002e	// [[OFFSET4]]

# CHECK: Tag: DW_TAG_base_type
# CHECK-NEXT: DW_IDX_die_offset: 0x0000002a // [[OFFSET3]]

# CHECK: Tag: DW_TAG_base_type
# CHECK-NEXT: DW_IDX_type_unit
# CHECK-NEXT: DW_IDX_die_offset: 0x0000003e // [[OFFSET1]]

# CHECK: Tag: DW_TAG_class_type
# CHECK-NEXT: DW_IDX_type_unit
# CHECK-NEXT: DW_IDX_die_offset: 0x00000023 // [[OFFSET]]

# CHECK: Tag: DW_TAG_variable
# CHECK-NEXT: DW_IDX_die_offset: 0x0000001f // [[OFFSET2]]

## int foo;
## class C1 {
##   public:
##   int v1;
##   int v2;
## };
##
## C1 v1;

	.file	"main.cpp"
	.file	0 "/OneType" "main.cpp" md5 0x8d91b0a8f262b9a2beb41d887793f45e
	.section	.debug_info,"G",@progbits,5175753803584721444,comdat
.Ltu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	2                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.quad	5175753803584721444             # Type Signature
	.long	35                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x18:0x2b DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.byte	2                               # Abbrev [2] 0x23:0x1b DW_TAG_class_type
	.byte	5                               # DW_AT_calling_convention
	.byte	8                               # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x29:0xa DW_TAG_member
	.byte	6                               # DW_AT_name
	.long	62                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	1                               # DW_AT_accessibility
                                        # DW_ACCESS_public
	.byte	3                               # Abbrev [3] 0x33:0xa DW_TAG_member
	.byte	7                               # DW_AT_name
	.long	62                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	4                               # DW_AT_data_member_location
	.byte	1                               # DW_AT_accessibility
                                        # DW_ACCESS_public
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x3e:0x4 DW_TAG_base_type
	.byte	5                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.type	foo,@object                     # @foo
	.bss
	.globl	foo
	.p2align	2, 0x0
foo:
	.long	0                               # 0x0
	.size	foo, 4

	.type	v1,@object                      # @v1
	.globl	v1
	.p2align	2, 0x0
v1:
	.zero	8
	.size	v1, 8

	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	65                              # DW_TAG_type_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	114                             # DW_AT_str_offsets_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	2                               # DW_TAG_class_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	13                              # DW_TAG_member
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	56                              # DW_AT_data_member_location
	.byte	11                              # DW_FORM_data1
	.byte	50                              # DW_AT_accessibility
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	37                              # DW_FORM_strx1
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.ascii	"\202|"                         # DW_AT_LLVM_sysroot
	.byte	37                              # DW_FORM_strx1
	.byte	114                             # DW_AT_str_offsets_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	37                              # DW_FORM_strx1
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	2                               # DW_TAG_class_type
	.byte	0                               # DW_CHILDREN_no
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	105                             # DW_AT_signature
	.byte	32                              # DW_FORM_ref_sig8
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	5                               # Abbrev [5] 0xc:0x37 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.byte	2                               # DW_AT_LLVM_sysroot
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	3                               # DW_AT_comp_dir
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	6                               # Abbrev [6] 0x1f:0xb DW_TAG_variable
	.byte	4                               # DW_AT_name
	.long	42                              # DW_AT_type
                                        # DW_AT_external
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	161
	.byte	0
	.byte	4                               # Abbrev [4] 0x2a:0x4 DW_TAG_base_type
	.byte	5                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	6                               # Abbrev [6] 0x2e:0xb DW_TAG_variable
	.byte	6                               # DW_AT_name
	.long	57                              # DW_AT_type
                                        # DW_AT_external
	.byte	0                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	161
	.byte	1
	.byte	7                               # Abbrev [7] 0x39:0x9 DW_TAG_class_type
                                        # DW_AT_declaration
	.quad	5175753803584721444             # DW_AT_signature
	.byte	0                               # End Of Children Mark
.Ldebug_info_end1:
	.section	.debug_str_offsets,"",@progbits
	.long	40                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 20.0.0git (git@github.com:llvm/llvm-project.git 4312075efa02ad861db0a19a0db8e6003aa06965)" # string offset=0
.Linfo_string1:
	.asciz	"main.cpp"                      # string offset=104
.Linfo_string2:
	.asciz	"/"                             # string offset=113
.Linfo_string3:
	.asciz	"/OneType" # string offset=115
.Linfo_string4:
	.asciz	"foo"                           # string offset=161
.Linfo_string5:
	.asciz	"int"                           # string offset=165
.Linfo_string6:
	.asciz	"v1"                            # string offset=169
.Linfo_string7:
	.asciz	"v2"                            # string offset=172
.Linfo_string8:
	.asciz	"C1"                            # string offset=175
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string6
	.long	.Linfo_string7
	.long	.Linfo_string8
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	foo
	.quad	v1
.Ldebug_addr_end0:
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
	.short	5                               # Header: version
	.short	0                               # Header: padding
	.long	1                               # Header: compilation unit count
	.long	1                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	4                               # Header: bucket count
	.long	4                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	.Ltu_begin0                     # Type unit 0
	.long	1                               # Bucket 0
	.long	3                               # Bucket 1
	.long	0                               # Bucket 2
	.long	0                               # Bucket 3
	.long	5863852                         # Hash in Bucket 0
	.long	193495088                       # Hash in Bucket 0
	.long	5863225                         # Hash in Bucket 1
	.long	193491849                       # Hash in Bucket 1
	.long	.Linfo_string6                  # String in Bucket 0: v1
	.long	.Linfo_string5                  # String in Bucket 0: int
	.long	.Linfo_string8                  # String in Bucket 1: C1
	.long	.Linfo_string4                  # String in Bucket 1: foo
	.long	.Lnames3-.Lnames_entries0       # Offset in Bucket 0
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 0
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 1
.Lnames_abbrev_start0:
	.byte	1                               # Abbrev code
	.byte	52                              # DW_TAG_variable
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	2                               # Abbrev code
	.byte	36                              # DW_TAG_base_type
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	3                               # Abbrev code
	.byte	36                              # DW_TAG_base_type
	.byte	2                               # DW_IDX_type_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	4                               # Abbrev code
	.byte	2                               # DW_TAG_class_type
	.byte	2                               # DW_IDX_type_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames3:
.L4:
	.byte	1                               # Abbreviation code
	.long	46                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: v1
.Lnames0:
.L0:
	.byte	2                               # Abbreviation code
	.long	42                              # DW_IDX_die_offset
.L2:                                    # DW_IDX_parent
	.byte	3                               # Abbreviation code
	.byte	0                               # DW_IDX_type_unit
	.long	62                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: int
.Lnames2:
.L3:
	.byte	4                               # Abbreviation code
	.byte	0                               # DW_IDX_type_unit
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: C1
.Lnames1:
.L1:
	.byte	1                               # Abbreviation code
	.long	31                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: foo
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 20.0.0git (git@github.com:llvm/llvm-project.git 4312075efa02ad861db0a19a0db8e6003aa06965)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
