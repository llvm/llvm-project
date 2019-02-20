# RUN: llvm-mc < %s -filetype obj -triple i386-pc-linux -o - \
# RUN:   | llvm-dwarfdump - | FileCheck %s

# CHECK: DW_TAG_variable

# base_type
# CHECK:   DW_AT_type{{.*}}"int"

# pointer_type
# CHECK:   DW_AT_type{{.*}}"int*"

# reference_type
# CHECK:   DW_AT_type{{.*}}"int&"

# rvalue_reference_type
# CHECK:   DW_AT_type{{.*}}"int&&"

# ptr_to_member_type
# CHECK:   DW_AT_type{{.*}}"int foo::*"

# array_type
# CHECK:   DW_AT_type{{.*}}"int
# Testing with a default lower bound of 0 and the following explicit bounds:
#   lower_bound(1)
# CHECK-NOT: {{.}}
# CHECK-SAME: {{\[}}[1, ?)]
#   upper_bound(2)
# CHECK-NOT: {{.}}
# CHECK-SAME: [3]
#   lower(1) and upper(2)
# CHECK-NOT: {{.}}
# CHECK-SAME: {{\[}}[1, 3)]
#   lower(1) and count(3)
# CHECK-NOT: {{.}}
# CHECK-SAME: {{\[}}[1, 4)]
#   lower(0) and count(4) - testing that the lower bound matching language
#   default is not rendered
# CHECK-NOT: {{.}}
# CHECK-SAME: [4]
#   count(2)
# CHECK-SAME: [2]
#   no attributes
# CHECK-NOT: {{.}}
# CHECK-SAME: []{{"\)$}}


# subroutine types
# CHECK:   DW_AT_type{{.*}}"int()"
# CHECK:   DW_AT_type{{.*}}"void(int)"
# CHECK:   DW_AT_type{{.*}}"void(int, int)"

# array_type with a language with a default lower bound of 1 instead of 0 and
# an upper bound of 2. This describes an array with 2 elements (whereas with a
# default lower bound of 0 it would be an array of 3 elements)
# CHECK: DW_AT_type{{.*}}"int[2]"

	.section	.debug_str,"MS",@progbits,1
.Lint_name:
	.asciz	"int"
.Lfoo_name:
	.asciz	"foo"
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	36                      # DW_TAG_base_type
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	3                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	4                       # Abbreviation Code
	.byte	0xf                     # DW_TAG_pointer_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	5                       # Abbreviation Code
	.byte	0x10                    # DW_TAG_reference_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	6                       # Abbreviation Code
	.byte	0x42                    # DW_TAG_rvalue_reference_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	7                       # Abbreviation Code
	.byte	0x1f                    # DW_TAG_ptr_to_member_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0x1d                    # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	8                       # Abbreviation Code
	.byte	1                       # DW_TAG_array_type
	.byte	1                       # DW_CHILDREN_yes
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	9                       # Abbreviation Code
	.byte	0x21                    # DW_TAG_subrange_type
	.byte	0                       # DW_CHILDREN_no
	.byte	0x22                    # DW_AT_lower_bound
	.byte	0xb                     # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	10                      # Abbreviation Code
	.byte	0x21                    # DW_TAG_subrange_type
	.byte	0                       # DW_CHILDREN_no
	.byte	0x2f                    # DW_AT_upper_bound
	.byte	0xb                     # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	11                      # Abbreviation Code
	.byte	0x21                    # DW_TAG_subrange_type
	.byte	0                       # DW_CHILDREN_no
	.byte	0x22                    # DW_AT_lower_bound
	.byte	0xb                     # DW_FORM_data1
	.byte	0x2f                    # DW_AT_upper_bound
	.byte	0xb                     # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	12                      # Abbreviation Code
	.byte	0x21                    # DW_TAG_subrange_type
	.byte	0                       # DW_CHILDREN_no
	.byte	0x22                    # DW_AT_lower_bound
	.byte	0xb                     # DW_FORM_data1
	.byte	0x37                    # DW_AT_count
	.byte	0xb                     # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	13                      # Abbreviation Code
	.byte	0x21                    # DW_TAG_subrange_type
	.byte	0                       # DW_CHILDREN_no
	.byte	0x37                    # DW_AT_count
	.byte	0xb                     # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	14                      # Abbreviation Code
	.byte	0x13                    # DW_TAG_structure_type
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	15                      # Abbreviation Code
	.byte	0x15                    # DW_TAG_subroutine_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	16                      # Abbreviation Code
	.byte	0x15                    # DW_TAG_subroutine_type
	.byte	1                       # DW_CHILDREN_yes
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	17                      # Abbreviation Code
	.byte	0x5                     # DW_TAG_formal_parameter
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	18                      # Abbreviation Code
	.byte	0x21                    # DW_TAG_subrange_type
	.byte	0                       # DW_CHILDREN_no
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin:
	.long	.Lunit_end - .Lunit_start # Length of Unit
.Lunit_start:
	.short	4                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # DW_TAG_compile_unit
	.short	12                      #   DW_AT_language
.Lint_type:
	.byte	2                       # DW_TAG_base_type
	.long	.Lint_name              #   DW_AT_name
.Lpointer_type:
	.byte	4                       # DW_TAG_pointer_type
	.long	.Lint_type - .Lcu_begin #   DW_AT_type
.Lreference_type:
	.byte	5                       # DW_TAG_reference_type
	.long	.Lint_type - .Lcu_begin #   DW_AT_type
.Lrvalue_reference_type:
	.byte	6                       # DW_TAG_rvalue_reference_type
	.long	.Lint_type - .Lcu_begin #   DW_AT_type
.Lstruct_type:
	.byte	14			# DW_TAG_structure_type
	.long	.Lfoo_name              #   DW_AT_name
.Lptr_to_member_type:
	.byte	7                       # DW_TAG_ptr_to_member_type
	.long	.Lint_type - .Lcu_begin #   DW_AT_type
	.long   .Lstruct_type - .Lcu_begin #   DW_AT_containing_type
.Larray_type:
	.byte	8                       # DW_TAG_array_type
	.long	.Lint_type - .Lcu_begin #   DW_AT_type
	.byte	9                       #   DW_AT_subrange_type
	.byte   1                       #     DW_AT_lower_bound
	.byte	10                      #   DW_AT_subrange_type
	.byte   2                       #     DW_AT_upper_bound
	.byte	11                      #   DW_AT_subrange_type
	.byte   1                       #     DW_AT_lower_bound
	.byte   2                       #     DW_AT_upper_bound
	.byte	12                      #   DW_AT_subrange_type
	.byte   1                       #     DW_AT_lower_bound
	.byte   3                       #     DW_AT_count
	.byte	12                      #   DW_AT_subrange_type
	.byte   0                       #     DW_AT_lower_bound
	.byte   4                       #     DW_AT_count
	.byte	13                      #   DW_AT_subrange_type
	.byte   2                       #     DW_AT_count
	.byte	18                      #   DW_AT_subrange_type
	.byte	0                       # End Of Children Mark
.Lsub_int_empty_type:
	.byte	15                      # DW_TAG_subroutine_type
	.long	.Lint_type - .Lcu_begin #   DW_AT_type
.Lsub_void_int_type:
	.byte	16                      # DW_TAG_subroutine_type
	.byte	17                       #   DW_TAG_formal_parameter
	.long	.Lint_type - .Lcu_begin #     DW_AT_type
	.byte	0                       # End Of Children Mark
.Lsub_void_int_int_type:
	.byte	16                      # DW_TAG_subroutine_type
	.byte	17                       #   DW_TAG_formal_parameter
	.long	.Lint_type - .Lcu_begin #     DW_AT_type
	.byte	17                       #   DW_TAG_formal_parameter
	.long	.Lint_type - .Lcu_begin #     DW_AT_type
	.byte	0                       # End Of Children Mark

	.byte	3                       # DW_TAG_variable
	.long	.Lint_type - .Lcu_begin #   DW_AT_type
	.byte	3                       # DW_TAG_variable
	.long	.Lpointer_type - .Lcu_begin #   DW_AT_type
	.byte	3                       # DW_TAG_variable
	.long	.Lreference_type - .Lcu_begin #   DW_AT_type
	.byte	3                       # DW_TAG_variable
	.long	.Lrvalue_reference_type - .Lcu_begin #   DW_AT_type
	.byte	3                       # DW_TAG_variable
	.long	.Lptr_to_member_type - .Lcu_begin #   DW_AT_type
	.byte	3                       # DW_TAG_variable
	.long	.Larray_type - .Lcu_begin #   DW_AT_type
	.byte	3                       # DW_TAG_variable
	.long	.Lsub_int_empty_type - .Lcu_begin #   DW_AT_type
	.byte	3                       # DW_TAG_variable
	.long	.Lsub_void_int_type - .Lcu_begin #   DW_AT_type
	.byte	3                       # DW_TAG_variable
	.long	.Lsub_void_int_int_type - .Lcu_begin #   DW_AT_type
	.byte	0                       # End Of Children Mark
.Lunit_end:
.Lcu2_begin:
	.long	.Lcu2_unit_end - .Lcu2_unit_start # Length of Unit
.Lcu2_unit_start:
	.short	4                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # DW_TAG_compile_unit
	.short	13                      #   DW_AT_language
.Lcu2_int_type:
	.byte	2                       # DW_TAG_base_type
	.long	.Lint_name              #   DW_AT_name
.Lcu2_array_type:
	.byte	8                       # DW_TAG_array_type
	.long	.Lcu2_int_type - .Lcu2_begin #   DW_AT_type
	.byte	10                      #   DW_AT_subrange_type
	.byte   2                       #     DW_AT_upper_bound
	.byte	3                       # DW_TAG_variable
	.long	.Lcu2_array_type - .Lcu2_begin #   DW_AT_type
.Lcu2_unit_end:
