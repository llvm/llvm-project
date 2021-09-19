# RUN: llvm-mc < %s -filetype obj -triple x86_64 -o - \
# RUN:   | llvm-dwarfdump --name=t1 --show-children - | FileCheck %s

# Assembly generated from the following source:
# struct foo;
# template<typename ...Ts>
# struct t1 { };
# t1<int, int*, int&, int&&, int foo::*, void(foo::*)(int), int (), void (int), void (int, int), void (*)(foo*, int)> v1;
#
# With the name of the "t1<...>" type renamed to "t1" (or using
# -gsimple-template-names) to make it easy to query for.
# Note that the llvm-dwarfdump command uses --name=t1 --show-children to only
# dump the template type parameters, making it easy to order the types as
# intended and to avoid visiting subtypes that aren't intended to be tested
# separately.

# CHECK: DW_TAG_GNU_template_parameter_pack

# base_type
# CHECK:   DW_AT_type{{.*}}"int"

# pointer_type
# CHECK:   DW_AT_type{{.*}}"int *"

# reference_type
# CHECK:   DW_AT_type{{.*}}"int &"

# rvalue_reference_type
# CHECK:   DW_AT_type{{.*}}"int &&"

# ptr_to_member_type
# CHECK:   DW_AT_type{{.*}}"int foo::*"

# ptr_to_member_type to a member function
# CHECK:   DW_AT_type{{.*}}"void (foo::*)(int)"

# subroutine types
# CHECK:   DW_AT_type{{.*}}"int ()"
# CHECK:   DW_AT_type{{.*}}"void (int)"
# CHECK:   DW_AT_type{{.*}}"void (int, int)"
# CHECK:   DW_AT_type{{.*}}"void (*)(foo *, int)"

	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	14                              # DW_FORM_strp
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
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
	.byte	3                               # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.ascii	"\207\202\001"                  # DW_TAG_GNU_template_parameter_pack
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	16                              # DW_TAG_reference_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	66                              # DW_TAG_rvalue_reference_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.byte	31                              # DW_TAG_ptr_to_member_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	29                              # DW_AT_containing_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	12                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	13                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	52                              # DW_AT_artificial
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	14                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	15                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0xd7 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	33                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.byte	2                               # Abbrev [2] 0x1e:0x15 DW_TAG_variable
	.long	.Linfo_string3                  # DW_AT_name
	.long	51                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	v1
	.byte	3                               # Abbrev [3] 0x33:0x42 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string7                  # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	4                               # Abbrev [4] 0x3c:0x38 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string4                  # DW_AT_name
	.byte	5                               # Abbrev [5] 0x41:0x5 DW_TAG_template_type_parameter
	.long	117                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x46:0x5 DW_TAG_template_type_parameter
	.long	124                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x4b:0x5 DW_TAG_template_type_parameter
	.long	129                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x50:0x5 DW_TAG_template_type_parameter
	.long	134                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x55:0x5 DW_TAG_template_type_parameter
	.long	139                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x5a:0x5 DW_TAG_template_type_parameter
	.long	153                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x5f:0x5 DW_TAG_template_type_parameter
	.long	179                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x64:0x5 DW_TAG_template_type_parameter
	.long	184                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x69:0x5 DW_TAG_template_type_parameter
	.long	191                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x6e:0x5 DW_TAG_template_type_parameter
	.long	203                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x75:0x7 DW_TAG_base_type
	.long	.Linfo_string5                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	7                               # Abbrev [7] 0x7c:0x5 DW_TAG_pointer_type
	.long	117                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0x81:0x5 DW_TAG_reference_type
	.long	117                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0x86:0x5 DW_TAG_rvalue_reference_type
	.long	117                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x8b:0x9 DW_TAG_ptr_to_member_type
	.long	117                             # DW_AT_type
	.long	148                             # DW_AT_containing_type
	.byte	11                              # Abbrev [11] 0x94:0x5 DW_TAG_structure_type
	.long	.Linfo_string6                  # DW_AT_name
                                        # DW_AT_declaration
	.byte	10                              # Abbrev [10] 0x99:0x9 DW_TAG_ptr_to_member_type
	.long	162                             # DW_AT_type
	.long	148                             # DW_AT_containing_type
	.byte	12                              # Abbrev [12] 0xa2:0xc DW_TAG_subroutine_type
	.byte	13                              # Abbrev [13] 0xa3:0x5 DW_TAG_formal_parameter
	.long	174                             # DW_AT_type
                                        # DW_AT_artificial
	.byte	14                              # Abbrev [14] 0xa8:0x5 DW_TAG_formal_parameter
	.long	117                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	7                               # Abbrev [7] 0xae:0x5 DW_TAG_pointer_type
	.long	148                             # DW_AT_type
	.byte	15                              # Abbrev [15] 0xb3:0x5 DW_TAG_subroutine_type
	.long	117                             # DW_AT_type
	.byte	12                              # Abbrev [12] 0xb8:0x7 DW_TAG_subroutine_type
	.byte	14                              # Abbrev [14] 0xb9:0x5 DW_TAG_formal_parameter
	.long	117                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xbf:0xc DW_TAG_subroutine_type
	.byte	14                              # Abbrev [14] 0xc0:0x5 DW_TAG_formal_parameter
	.long	117                             # DW_AT_type
	.byte	14                              # Abbrev [14] 0xc5:0x5 DW_TAG_formal_parameter
	.long	117                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	7                               # Abbrev [7] 0xcb:0x5 DW_TAG_pointer_type
	.long	208                             # DW_AT_type
	.byte	12                              # Abbrev [12] 0xd0:0xc DW_TAG_subroutine_type
	.byte	14                              # Abbrev [14] 0xd1:0x5 DW_TAG_formal_parameter
	.long	220                             # DW_AT_type
	.byte	14                              # Abbrev [14] 0xd6:0x5 DW_TAG_formal_parameter
	.long	117                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	7                               # Abbrev [7] 0xdc:0x5 DW_TAG_pointer_type
	.long	148                             # DW_AT_type
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 14.0.0 (git@github.com:llvm/llvm-project.git 0543d3a279346152e88fb40f0f817ca8bd145864)" # string offset=0
.Linfo_string1:
	.asciz	"/usr/local/google/home/blaikie/dev/scratch/test.cpp" # string offset=101
.Linfo_string2:
	.asciz	"/usr/local/google/home/blaikie/dev/llvm/src" # string offset=153
.Linfo_string3:
	.asciz	"v1"                            # string offset=197
.Linfo_string4:
	.asciz	"Ts"                            # string offset=200
.Linfo_string5:
	.asciz	"int"                           # string offset=203
.Linfo_string6:
	.asciz	"foo"                           # string offset=207
.Linfo_string7:
	.asciz	"t1" 				# string offset=211
	.ident	"clang version 14.0.0 (git@github.com:llvm/llvm-project.git 0543d3a279346152e88fb40f0f817ca8bd145864)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
