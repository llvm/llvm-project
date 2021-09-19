# RUN: llvm-mc < %s -filetype obj -triple x86_64 -o - \
# RUN:   | llvm-dwarfdump --name=t1 --show-children - | FileCheck %s

# Assembly generated from the following source:
# 
#  struct foo;
#  template <typename... Ts>
#  struct t1 {};
#  t1<
#      // base type
#      int,
#      // reference type
#      int &,
#      // rvalue reference type
#      int &&,
#      // pointer types
#      int *, const void *const *, const void *const, int *const volatile,
#      int *volatile, void *const,
#      // pointer to member variable
#      int foo::*,
#      // pointer to member functions
#      void (foo::*)(int), void (foo::*const &)() const volatile &&,
#      // arrays
#      int *const (&)[1], int *const[1], const int (&)[1], const int[1],
#      // subroutine types
#      int(), void(int), void(int, int), void (*)(foo *, int), void (*const)(),
#      void() const, void() volatile &&, void() const volatile &,
#      void(const volatile foo *), void (*(int))(float)>
#      v1;
#
# With the name of the "t1<...>" type renamed to "t1" (or using
# -gsimple-template-names) to make it easy to query for.
# Note that the llvm-dwarfdump command uses --name=t1 --show-children to only
# dump the template type parameters, making it easy to order the types as
# intended and to avoid visiting subtypes that aren't intended to be tested
# separately.

# base_type
# CHECK:   DW_AT_type{{.*}}"int"

# reference_type
# CHECK:   DW_AT_type{{.*}}"int &"

# rvalue_reference_type
# CHECK:   DW_AT_type{{.*}}"int &&"

# pointer_type
# CHECK:   DW_AT_type{{.*}}"int *"
# CHECK:   DW_AT_type{{.*}}"const void *const *")
# CHECK:   DW_AT_type{{.*}}"const void *const")
# CHECK:   DW_AT_type{{.*}}"int *const volatile")
# CHECK:   DW_AT_type{{.*}}"int *volatile")
# CHECK:   DW_AT_type{{.*}}"void *const")

# ptr_to_member_type
# CHECK:   DW_AT_type{{.*}}"int foo::*"

# ptr_to_member_type to a member function
# CHECK:   DW_AT_type{{.*}}"void (foo::*)(int)"
# const reference to a pointer to member function (with const, volatile, rvalue ref qualifiers)
# CHECK:   DW_AT_type{{.*}}"void (foo::*const &)() const volatile &&")

# CHECK:   DW_AT_type{{.*}}"int *const (&)[1]")
# CHECK:   DW_AT_type{{.*}}"int *const[1]")
# CHECK:   DW_AT_type{{.*}}"const int (&)[1]")
# CHECK:   DW_AT_type{{.*}}"const int [1]")

# subroutine types
# CHECK:   DW_AT_type{{.*}}"int ()"
# CHECK:   DW_AT_type{{.*}}"void (int)"
# CHECK:   DW_AT_type{{.*}}"void (int, int)"
# CHECK:   DW_AT_type{{.*}}"void (*)(foo *, int)"
# CHECK:   DW_AT_type{{.*}}"void (*const)()")
# CHECK:   DW_AT_type{{.*}}"void () const")
# CHECK:   DW_AT_type{{.*}}"void () volatile &&")
# CHECK:   DW_AT_type{{.*}}"void () const volatile &")
# CHECK:   DW_AT_type{{.*}}"void (const volatile foo *)")
# CHECK:   DW_AT_type{{.*}}"void (*(int))(float)")

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
	.byte	16                              # DW_TAG_reference_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	66                              # DW_TAG_rvalue_reference_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	12                              # Abbreviation Code
	.byte	53                              # DW_TAG_volatile_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	13                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	14                              # Abbreviation Code
	.byte	31                              # DW_TAG_ptr_to_member_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	29                              # DW_AT_containing_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	15                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	16                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	17                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	52                              # DW_AT_artificial
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	18                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	19                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	120                             # DW_AT_rvalue_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	20                              # Abbreviation Code
	.byte	1                               # DW_TAG_array_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	21                              # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	55                              # DW_AT_count
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	22                              # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	23                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	24                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	25                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	120                             # DW_AT_rvalue_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	26                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	119                             # DW_AT_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	27                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
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
	.byte	1                               # Abbrev [1] 0xb:0x1ee DW_TAG_compile_unit
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
	.byte	24                              # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	v1
	.byte	3                               # Abbrev [3] 0x33:0x92 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string9                  # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	4                               # Abbrev [4] 0x3c:0x88 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string4                  # DW_AT_name
	.byte	5                               # Abbrev [5] 0x41:0x5 DW_TAG_template_type_parameter
	.long	197                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x46:0x5 DW_TAG_template_type_parameter
	.long	204                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x4b:0x5 DW_TAG_template_type_parameter
	.long	209                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x50:0x5 DW_TAG_template_type_parameter
	.long	214                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x55:0x5 DW_TAG_template_type_parameter
	.long	219                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x5a:0x5 DW_TAG_template_type_parameter
	.long	224                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x5f:0x5 DW_TAG_template_type_parameter
	.long	235                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x64:0x5 DW_TAG_template_type_parameter
	.long	240                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x69:0x5 DW_TAG_template_type_parameter
	.long	245                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x6e:0x5 DW_TAG_template_type_parameter
	.long	251                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x73:0x5 DW_TAG_template_type_parameter
	.long	265                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x78:0x5 DW_TAG_template_type_parameter
	.long	291                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x7d:0x5 DW_TAG_template_type_parameter
	.long	332                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x82:0x5 DW_TAG_template_type_parameter
	.long	337                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x87:0x5 DW_TAG_template_type_parameter
	.long	361                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x8c:0x5 DW_TAG_template_type_parameter
	.long	366                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x91:0x5 DW_TAG_template_type_parameter
	.long	383                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x96:0x5 DW_TAG_template_type_parameter
	.long	388                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x9b:0x5 DW_TAG_template_type_parameter
	.long	395                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xa0:0x5 DW_TAG_template_type_parameter
	.long	407                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xa5:0x5 DW_TAG_template_type_parameter
	.long	429                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xaa:0x5 DW_TAG_template_type_parameter
	.long	440                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xaf:0x5 DW_TAG_template_type_parameter
	.long	445                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xb4:0x5 DW_TAG_template_type_parameter
	.long	451                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xb9:0x5 DW_TAG_template_type_parameter
	.long	462                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xbe:0x5 DW_TAG_template_type_parameter
	.long	474                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0xc5:0x7 DW_TAG_base_type
	.long	.Linfo_string5                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	7                               # Abbrev [7] 0xcc:0x5 DW_TAG_reference_type
	.long	197                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0xd1:0x5 DW_TAG_rvalue_reference_type
	.long	197                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0xd6:0x5 DW_TAG_pointer_type
	.long	197                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0xdb:0x5 DW_TAG_pointer_type
	.long	224                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0xe0:0x5 DW_TAG_const_type
	.long	229                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0xe5:0x5 DW_TAG_pointer_type
	.long	234                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0xea:0x1 DW_TAG_const_type
	.byte	10                              # Abbrev [10] 0xeb:0x5 DW_TAG_const_type
	.long	240                             # DW_AT_type
	.byte	12                              # Abbrev [12] 0xf0:0x5 DW_TAG_volatile_type
	.long	214                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0xf5:0x5 DW_TAG_const_type
	.long	250                             # DW_AT_type
	.byte	13                              # Abbrev [13] 0xfa:0x1 DW_TAG_pointer_type
	.byte	14                              # Abbrev [14] 0xfb:0x9 DW_TAG_ptr_to_member_type
	.long	197                             # DW_AT_type
	.long	260                             # DW_AT_containing_type
	.byte	15                              # Abbrev [15] 0x104:0x5 DW_TAG_structure_type
	.long	.Linfo_string6                  # DW_AT_name
                                        # DW_AT_declaration
	.byte	14                              # Abbrev [14] 0x109:0x9 DW_TAG_ptr_to_member_type
	.long	274                             # DW_AT_type
	.long	260                             # DW_AT_containing_type
	.byte	16                              # Abbrev [16] 0x112:0xc DW_TAG_subroutine_type
	.byte	17                              # Abbrev [17] 0x113:0x5 DW_TAG_formal_parameter
	.long	286                             # DW_AT_type
                                        # DW_AT_artificial
	.byte	18                              # Abbrev [18] 0x118:0x5 DW_TAG_formal_parameter
	.long	197                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	9                               # Abbrev [9] 0x11e:0x5 DW_TAG_pointer_type
	.long	260                             # DW_AT_type
	.byte	7                               # Abbrev [7] 0x123:0x5 DW_TAG_reference_type
	.long	296                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x128:0x5 DW_TAG_const_type
	.long	301                             # DW_AT_type
	.byte	14                              # Abbrev [14] 0x12d:0x9 DW_TAG_ptr_to_member_type
	.long	310                             # DW_AT_type
	.long	260                             # DW_AT_containing_type
	.byte	19                              # Abbrev [19] 0x136:0x7 DW_TAG_subroutine_type
                                        # DW_AT_rvalue_reference
	.byte	17                              # Abbrev [17] 0x137:0x5 DW_TAG_formal_parameter
	.long	317                             # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	9                               # Abbrev [9] 0x13d:0x5 DW_TAG_pointer_type
	.long	322                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x142:0x5 DW_TAG_const_type
	.long	327                             # DW_AT_type
	.byte	12                              # Abbrev [12] 0x147:0x5 DW_TAG_volatile_type
	.long	260                             # DW_AT_type
	.byte	7                               # Abbrev [7] 0x14c:0x5 DW_TAG_reference_type
	.long	337                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x151:0x5 DW_TAG_const_type
	.long	342                             # DW_AT_type
	.byte	20                              # Abbrev [20] 0x156:0xc DW_TAG_array_type
	.long	214                             # DW_AT_type
	.byte	21                              # Abbrev [21] 0x15b:0x6 DW_TAG_subrange_type
	.long	354                             # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	22                              # Abbrev [22] 0x162:0x7 DW_TAG_base_type
	.long	.Linfo_string7                  # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	7                               # DW_AT_encoding
	.byte	7                               # Abbrev [7] 0x169:0x5 DW_TAG_reference_type
	.long	366                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x16e:0x5 DW_TAG_const_type
	.long	371                             # DW_AT_type
	.byte	20                              # Abbrev [20] 0x173:0xc DW_TAG_array_type
	.long	197                             # DW_AT_type
	.byte	21                              # Abbrev [21] 0x178:0x6 DW_TAG_subrange_type
	.long	354                             # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x17f:0x5 DW_TAG_subroutine_type
	.long	197                             # DW_AT_type
	.byte	16                              # Abbrev [16] 0x184:0x7 DW_TAG_subroutine_type
	.byte	18                              # Abbrev [18] 0x185:0x5 DW_TAG_formal_parameter
	.long	197                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0x18b:0xc DW_TAG_subroutine_type
	.byte	18                              # Abbrev [18] 0x18c:0x5 DW_TAG_formal_parameter
	.long	197                             # DW_AT_type
	.byte	18                              # Abbrev [18] 0x191:0x5 DW_TAG_formal_parameter
	.long	197                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	9                               # Abbrev [9] 0x197:0x5 DW_TAG_pointer_type
	.long	412                             # DW_AT_type
	.byte	16                              # Abbrev [16] 0x19c:0xc DW_TAG_subroutine_type
	.byte	18                              # Abbrev [18] 0x19d:0x5 DW_TAG_formal_parameter
	.long	424                             # DW_AT_type
	.byte	18                              # Abbrev [18] 0x1a2:0x5 DW_TAG_formal_parameter
	.long	197                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	9                               # Abbrev [9] 0x1a8:0x5 DW_TAG_pointer_type
	.long	260                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x1ad:0x5 DW_TAG_const_type
	.long	434                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0x1b2:0x5 DW_TAG_pointer_type
	.long	439                             # DW_AT_type
	.byte	24                              # Abbrev [24] 0x1b7:0x1 DW_TAG_subroutine_type
	.byte	10                              # Abbrev [10] 0x1b8:0x5 DW_TAG_const_type
	.long	439                             # DW_AT_type
	.byte	12                              # Abbrev [12] 0x1bd:0x5 DW_TAG_volatile_type
	.long	450                             # DW_AT_type
	.byte	25                              # Abbrev [25] 0x1c2:0x1 DW_TAG_subroutine_type
                                        # DW_AT_rvalue_reference
	.byte	10                              # Abbrev [10] 0x1c3:0x5 DW_TAG_const_type
	.long	456                             # DW_AT_type
	.byte	12                              # Abbrev [12] 0x1c8:0x5 DW_TAG_volatile_type
	.long	461                             # DW_AT_type
	.byte	26                              # Abbrev [26] 0x1cd:0x1 DW_TAG_subroutine_type
                                        # DW_AT_reference
	.byte	16                              # Abbrev [16] 0x1ce:0x7 DW_TAG_subroutine_type
	.byte	18                              # Abbrev [18] 0x1cf:0x5 DW_TAG_formal_parameter
	.long	469                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	9                               # Abbrev [9] 0x1d5:0x5 DW_TAG_pointer_type
	.long	322                             # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1da:0xb DW_TAG_subroutine_type
	.long	485                             # DW_AT_type
	.byte	18                              # Abbrev [18] 0x1df:0x5 DW_TAG_formal_parameter
	.long	197                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	9                               # Abbrev [9] 0x1e5:0x5 DW_TAG_pointer_type
	.long	490                             # DW_AT_type
	.byte	16                              # Abbrev [16] 0x1ea:0x7 DW_TAG_subroutine_type
	.byte	18                              # Abbrev [18] 0x1eb:0x5 DW_TAG_formal_parameter
	.long	497                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x1f1:0x7 DW_TAG_base_type
	.long	.Linfo_string8                  # DW_AT_name
	.byte	4                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 14.0.0 (git@github.com:llvm/llvm-project.git 0543d3a279346152e88fb40f0f817ca8bd145864)" # string offset=0
.Linfo_string1:
	.asciz	"test.cpp"                      # string offset=101
.Linfo_string2:
	.asciz	"/usr/local/google/home/blaikie/dev/scratch" # string offset=110
.Linfo_string3:
	.asciz	"v1"                            # string offset=153
.Linfo_string4:
	.asciz	"Ts"                            # string offset=156
.Linfo_string5:
	.asciz	"int"                           # string offset=159
.Linfo_string6:
	.asciz	"foo"                           # string offset=163
.Linfo_string7:
	.asciz	"__ARRAY_SIZE_TYPE__"           # string offset=167
.Linfo_string8:
	.asciz	"float"                         # string offset=187
.Linfo_string9:
	.asciz	"t1"                            # string offset=193
	.ident	"clang version 14.0.0 (git@github.com:llvm/llvm-project.git 0543d3a279346152e88fb40f0f817ca8bd145864)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
