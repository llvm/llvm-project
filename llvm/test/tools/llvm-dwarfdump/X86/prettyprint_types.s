# RUN: llvm-mc < %s -filetype obj -triple x86_64 -o - \
# RUN:   | llvm-dwarfdump --name=t1 --show-children - | FileCheck %s

# Assembly generated from the following source:
# 
#  struct foo;
#  template <typename... Ts>
#  struct t1 {};
#  namespace ns {
#  struct inner { };
#  }
#  namespace {
#  struct anon_ns_mem { };
#  }
#  t1<
#      // base type
#      int,
#      // nullptr unspecified type
#      decltype(nullptr),
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
#      void(const volatile foo *), void (*(int))(float),
#      // qualified types
#      ns::inner, ns::inner(), ns::inner[1], ns::inner *, ns::inner ns::inner::*,
#      const ns::inner, anon_ns_mem>
#      v1;
#  // extern function to force the code to be emitted - otherwise v1 (having
#  // internal linkage due to being of a type instantiated with an internal
#  // linkage type) would be optimized away as unused.
#  __attribute__((nodebug)) void*f2() {
#    return &v1;
#  }
# 
# With the name of the "t1<...>" type renamed to "t1" (or using
# -gsimple-template-names) to make it easy to query for.
# Note that the llvm-dwarfdump command uses --name=t1 --show-children to only
# dump the template type parameters, making it easy to order the types as
# intended and to avoid visiting subtypes that aren't intended to be tested
# separately.

# base_type
# CHECK:   DW_AT_type{{.*}}"int"

# FIXME: This might change in clang to std::nullptr_t which would be more
# accurate and not be ambiguous with some arbitrary ::nullptr_t a user could
# define. If that change is made, this code/test should be fixed too.
# nullptr_t unspecified type
# CHECK:   DW_AT_type{{.*}}"std::nullptr_t"

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

# qualified types
# CHECK:   DW_AT_type{{.*}}"ns::inner"
# CHECK:   DW_AT_type{{.*}}"ns::inner ()"
# CHECK:   DW_AT_type{{.*}}"ns::inner [1]"
# CHECK:   DW_AT_type{{.*}}"ns::inner *"
# CHECK:   DW_AT_type{{.*}}"ns::inner (ns::inner::*)(ns::inner)"
# CHECK:   DW_AT_type{{.*}}"const ns::inner"
# CHECK:   DW_AT_type{{.*}}"(anonymous namespace)::anon_ns_mem"

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
	.byte	59                              # DW_TAG_unspecified_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
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
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	12                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	13                              # Abbreviation Code
	.byte	53                              # DW_TAG_volatile_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	14                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	15                              # Abbreviation Code
	.byte	31                              # DW_TAG_ptr_to_member_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	29                              # DW_AT_containing_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	16                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	17                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	18                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	52                              # DW_AT_artificial
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	19                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	20                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	120                             # DW_AT_rvalue_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	21                              # Abbreviation Code
	.byte	1                               # DW_TAG_array_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	22                              # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	55                              # DW_AT_count
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	23                              # Abbreviation Code
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
	.byte	24                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	25                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	26                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	120                             # DW_AT_rvalue_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	27                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	119                             # DW_AT_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	28                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	29                              # Abbreviation Code
	.byte	57                              # DW_TAG_namespace
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	30                              # Abbreviation Code
	.byte	57                              # DW_TAG_namespace
	.byte	1                               # DW_CHILDREN_yes
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
	.byte	1                               # Abbrev [1] 0xb:0x266 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	33                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.byte	2                               # Abbrev [2] 0x1e:0x15 DW_TAG_variable
	.long	.Linfo_string3                  # DW_AT_name
	.long	51                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	35                              # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	v1
	.byte	3                               # Abbrev [3] 0x33:0xba DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string13                 # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	4                               # Abbrev [4] 0x3c:0xb0 DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string4                  # DW_AT_name
	.byte	5                               # Abbrev [5] 0x41:0x5 DW_TAG_template_type_parameter
	.long	237                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x46:0x5 DW_TAG_template_type_parameter
	.long	244                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x4b:0x5 DW_TAG_template_type_parameter
	.long	249                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x50:0x5 DW_TAG_template_type_parameter
	.long	254                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x55:0x5 DW_TAG_template_type_parameter
	.long	259                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x5a:0x5 DW_TAG_template_type_parameter
	.long	264                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x5f:0x5 DW_TAG_template_type_parameter
	.long	269                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x64:0x5 DW_TAG_template_type_parameter
	.long	280                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x69:0x5 DW_TAG_template_type_parameter
	.long	285                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x6e:0x5 DW_TAG_template_type_parameter
	.long	290                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x73:0x5 DW_TAG_template_type_parameter
	.long	296                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x78:0x5 DW_TAG_template_type_parameter
	.long	310                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x7d:0x5 DW_TAG_template_type_parameter
	.long	336                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x82:0x5 DW_TAG_template_type_parameter
	.long	377                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x87:0x5 DW_TAG_template_type_parameter
	.long	382                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x8c:0x5 DW_TAG_template_type_parameter
	.long	406                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x91:0x5 DW_TAG_template_type_parameter
	.long	411                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x96:0x5 DW_TAG_template_type_parameter
	.long	428                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x9b:0x5 DW_TAG_template_type_parameter
	.long	433                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xa0:0x5 DW_TAG_template_type_parameter
	.long	440                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xa5:0x5 DW_TAG_template_type_parameter
	.long	452                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xaa:0x5 DW_TAG_template_type_parameter
	.long	474                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xaf:0x5 DW_TAG_template_type_parameter
	.long	485                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xb4:0x5 DW_TAG_template_type_parameter
	.long	490                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xb9:0x5 DW_TAG_template_type_parameter
	.long	496                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xbe:0x5 DW_TAG_template_type_parameter
	.long	507                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xc3:0x5 DW_TAG_template_type_parameter
	.long	519                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xc8:0x5 DW_TAG_template_type_parameter
	.long	554                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xcd:0x5 DW_TAG_template_type_parameter
	.long	560                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xd2:0x5 DW_TAG_template_type_parameter
	.long	565                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xd7:0x5 DW_TAG_template_type_parameter
	.long	577                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xdc:0x5 DW_TAG_template_type_parameter
	.long	582                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xe1:0x5 DW_TAG_template_type_parameter
	.long	612                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xe6:0x5 DW_TAG_template_type_parameter
	.long	618                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0xed:0x7 DW_TAG_base_type
	.long	.Linfo_string5                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	7                               # Abbrev [7] 0xf4:0x5 DW_TAG_unspecified_type
	.long	.Linfo_string6                  # DW_AT_name
	.byte	8                               # Abbrev [8] 0xf9:0x5 DW_TAG_reference_type
	.long	237                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0xfe:0x5 DW_TAG_rvalue_reference_type
	.long	237                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x103:0x5 DW_TAG_pointer_type
	.long	237                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x108:0x5 DW_TAG_pointer_type
	.long	269                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x10d:0x5 DW_TAG_const_type
	.long	274                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x112:0x5 DW_TAG_pointer_type
	.long	279                             # DW_AT_type
	.byte	12                              # Abbrev [12] 0x117:0x1 DW_TAG_const_type
	.byte	11                              # Abbrev [11] 0x118:0x5 DW_TAG_const_type
	.long	285                             # DW_AT_type
	.byte	13                              # Abbrev [13] 0x11d:0x5 DW_TAG_volatile_type
	.long	259                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x122:0x5 DW_TAG_const_type
	.long	295                             # DW_AT_type
	.byte	14                              # Abbrev [14] 0x127:0x1 DW_TAG_pointer_type
	.byte	15                              # Abbrev [15] 0x128:0x9 DW_TAG_ptr_to_member_type
	.long	237                             # DW_AT_type
	.long	305                             # DW_AT_containing_type
	.byte	16                              # Abbrev [16] 0x131:0x5 DW_TAG_structure_type
	.long	.Linfo_string7                  # DW_AT_name
                                        # DW_AT_declaration
	.byte	15                              # Abbrev [15] 0x136:0x9 DW_TAG_ptr_to_member_type
	.long	319                             # DW_AT_type
	.long	305                             # DW_AT_containing_type
	.byte	17                              # Abbrev [17] 0x13f:0xc DW_TAG_subroutine_type
	.byte	18                              # Abbrev [18] 0x140:0x5 DW_TAG_formal_parameter
	.long	331                             # DW_AT_type
                                        # DW_AT_artificial
	.byte	19                              # Abbrev [19] 0x145:0x5 DW_TAG_formal_parameter
	.long	237                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x14b:0x5 DW_TAG_pointer_type
	.long	305                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0x150:0x5 DW_TAG_reference_type
	.long	341                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x155:0x5 DW_TAG_const_type
	.long	346                             # DW_AT_type
	.byte	15                              # Abbrev [15] 0x15a:0x9 DW_TAG_ptr_to_member_type
	.long	355                             # DW_AT_type
	.long	305                             # DW_AT_containing_type
	.byte	20                              # Abbrev [20] 0x163:0x7 DW_TAG_subroutine_type
                                        # DW_AT_rvalue_reference
	.byte	18                              # Abbrev [18] 0x164:0x5 DW_TAG_formal_parameter
	.long	362                             # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x16a:0x5 DW_TAG_pointer_type
	.long	367                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x16f:0x5 DW_TAG_const_type
	.long	372                             # DW_AT_type
	.byte	13                              # Abbrev [13] 0x174:0x5 DW_TAG_volatile_type
	.long	305                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0x179:0x5 DW_TAG_reference_type
	.long	382                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x17e:0x5 DW_TAG_const_type
	.long	387                             # DW_AT_type
	.byte	21                              # Abbrev [21] 0x183:0xc DW_TAG_array_type
	.long	259                             # DW_AT_type
	.byte	22                              # Abbrev [22] 0x188:0x6 DW_TAG_subrange_type
	.long	399                             # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x18f:0x7 DW_TAG_base_type
	.long	.Linfo_string8                  # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	7                               # DW_AT_encoding
	.byte	8                               # Abbrev [8] 0x196:0x5 DW_TAG_reference_type
	.long	411                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x19b:0x5 DW_TAG_const_type
	.long	416                             # DW_AT_type
	.byte	21                              # Abbrev [21] 0x1a0:0xc DW_TAG_array_type
	.long	237                             # DW_AT_type
	.byte	22                              # Abbrev [22] 0x1a5:0x6 DW_TAG_subrange_type
	.long	399                             # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	24                              # Abbrev [24] 0x1ac:0x5 DW_TAG_subroutine_type
	.long	237                             # DW_AT_type
	.byte	17                              # Abbrev [17] 0x1b1:0x7 DW_TAG_subroutine_type
	.byte	19                              # Abbrev [19] 0x1b2:0x5 DW_TAG_formal_parameter
	.long	237                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0x1b8:0xc DW_TAG_subroutine_type
	.byte	19                              # Abbrev [19] 0x1b9:0x5 DW_TAG_formal_parameter
	.long	237                             # DW_AT_type
	.byte	19                              # Abbrev [19] 0x1be:0x5 DW_TAG_formal_parameter
	.long	237                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1c4:0x5 DW_TAG_pointer_type
	.long	457                             # DW_AT_type
	.byte	17                              # Abbrev [17] 0x1c9:0xc DW_TAG_subroutine_type
	.byte	19                              # Abbrev [19] 0x1ca:0x5 DW_TAG_formal_parameter
	.long	469                             # DW_AT_type
	.byte	19                              # Abbrev [19] 0x1cf:0x5 DW_TAG_formal_parameter
	.long	237                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1d5:0x5 DW_TAG_pointer_type
	.long	305                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x1da:0x5 DW_TAG_const_type
	.long	479                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x1df:0x5 DW_TAG_pointer_type
	.long	484                             # DW_AT_type
	.byte	25                              # Abbrev [25] 0x1e4:0x1 DW_TAG_subroutine_type
	.byte	11                              # Abbrev [11] 0x1e5:0x5 DW_TAG_const_type
	.long	484                             # DW_AT_type
	.byte	13                              # Abbrev [13] 0x1ea:0x5 DW_TAG_volatile_type
	.long	495                             # DW_AT_type
	.byte	26                              # Abbrev [26] 0x1ef:0x1 DW_TAG_subroutine_type
                                        # DW_AT_rvalue_reference
	.byte	11                              # Abbrev [11] 0x1f0:0x5 DW_TAG_const_type
	.long	501                             # DW_AT_type
	.byte	13                              # Abbrev [13] 0x1f5:0x5 DW_TAG_volatile_type
	.long	506                             # DW_AT_type
	.byte	27                              # Abbrev [27] 0x1fa:0x1 DW_TAG_subroutine_type
                                        # DW_AT_reference
	.byte	17                              # Abbrev [17] 0x1fb:0x7 DW_TAG_subroutine_type
	.byte	19                              # Abbrev [19] 0x1fc:0x5 DW_TAG_formal_parameter
	.long	514                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x202:0x5 DW_TAG_pointer_type
	.long	367                             # DW_AT_type
	.byte	28                              # Abbrev [28] 0x207:0xb DW_TAG_subroutine_type
	.long	530                             # DW_AT_type
	.byte	19                              # Abbrev [19] 0x20c:0x5 DW_TAG_formal_parameter
	.long	237                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x212:0x5 DW_TAG_pointer_type
	.long	535                             # DW_AT_type
	.byte	17                              # Abbrev [17] 0x217:0x7 DW_TAG_subroutine_type
	.byte	19                              # Abbrev [19] 0x218:0x5 DW_TAG_formal_parameter
	.long	542                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x21e:0x7 DW_TAG_base_type
	.long	.Linfo_string9                  # DW_AT_name
	.byte	4                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	29                              # Abbrev [29] 0x225:0xb DW_TAG_namespace
	.long	.Linfo_string10                 # DW_AT_name
	.byte	16                              # Abbrev [16] 0x22a:0x5 DW_TAG_structure_type
	.long	.Linfo_string11                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	24                              # Abbrev [24] 0x230:0x5 DW_TAG_subroutine_type
	.long	554                             # DW_AT_type
	.byte	21                              # Abbrev [21] 0x235:0xc DW_TAG_array_type
	.long	554                             # DW_AT_type
	.byte	22                              # Abbrev [22] 0x23a:0x6 DW_TAG_subrange_type
	.long	399                             # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x241:0x5 DW_TAG_pointer_type
	.long	554                             # DW_AT_type
	.byte	15                              # Abbrev [15] 0x246:0x9 DW_TAG_ptr_to_member_type
	.long	591                             # DW_AT_type
	.long	554                             # DW_AT_containing_type
	.byte	28                              # Abbrev [28] 0x24f:0x10 DW_TAG_subroutine_type
	.long	554                             # DW_AT_type
	.byte	18                              # Abbrev [18] 0x254:0x5 DW_TAG_formal_parameter
	.long	607                             # DW_AT_type
                                        # DW_AT_artificial
	.byte	19                              # Abbrev [19] 0x259:0x5 DW_TAG_formal_parameter
	.long	554                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x25f:0x5 DW_TAG_pointer_type
	.long	554                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x264:0x5 DW_TAG_const_type
	.long	554                             # DW_AT_type
	.byte	30                              # Abbrev [30] 0x269:0x7 DW_TAG_namespace
	.byte	16                              # Abbrev [16] 0x26a:0x5 DW_TAG_structure_type
	.long	.Linfo_string12                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 14.0.0 (git@github.com:llvm/llvm-project.git 559ab58ca57486621857a630d16dcddb6cceedfb)" # string offset=0
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
	.asciz	"decltype(nullptr)"             # string offset=163
.Linfo_string7:
	.asciz	"foo"                           # string offset=181
.Linfo_string8:
	.asciz	"__ARRAY_SIZE_TYPE__"           # string offset=185
.Linfo_string9:
	.asciz	"float"                         # string offset=205
.Linfo_string10:
	.asciz	"ns"                            # string offset=211
.Linfo_string11:
	.asciz	"inner"                         # string offset=214
.Linfo_string12:
	.asciz	"anon_ns_mem"                   # string offset=220
.Linfo_string13:
	.asciz	"t1"                            # string offset=232
	.ident	"clang version 14.0.0 (git@github.com:llvm/llvm-project.git 559ab58ca57486621857a630d16dcddb6cceedfb)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym v1
	.section	.debug_line,"",@progbits
.Lline_table_start0:
