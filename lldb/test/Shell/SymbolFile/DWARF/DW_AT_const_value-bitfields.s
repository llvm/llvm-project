# RUN: llvm-mc -filetype=obj -o %t -triple x86_64-apple-macosx10.15.0 %s
# RUN: %lldb %t -o "target variable constant" -b | FileCheck %s

# CHECK: (lldb) target variable constant
# CHECK: (U) constant = {
# CHECK:   raw = 1688469761
# CHECK:    = (a = 1, b = 1, c = 36, d = 2, e = 36, f = 1)
# CHECK: }

# This is testing when how ValueObjectVariable handles the case where the
# DWARFExpression holds the data that represents a constant value.
# Compile at -O1 allows us to capture this case. Below is the code used
# to generate the assembly:
#
# typedef union
# {
#   unsigned raw;
#   struct
#   {
#     unsigned a : 8;
#     unsigned b : 8;
#     unsigned c : 6;
#     unsigned d : 2;
#     unsigned e : 6;
#     unsigned f : 2;
#   } ;
# } U;
#
# static U __attribute__((used)) _type_anchor;
# static const int constant = 0x64A40101;
#
# int g() { return constant; }
#
# int main() {
#   U u;
#   u.raw = 0x64A40101;
# }
#
# Compiled as follows:
#
#   clang -gdwarf-4 -O1 dw_at_const_value_bug.c -S -o dw_at_const_value_bug.s
#
# I was able to obtain a global of type U with DW_AT_const_value but was able
# to using int. This required modifying the DW_AT_type of constant to be type
# U. After that stripping as much of the assembly as possible to give us a
# smaller reproducer.


.zerofill __DATA,__bss,__type_anchor,4,2 ## @_type_anchor
	.no_dead_strip	__type_anchor
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
  .zero 90
	.asciz	"constant"              ## string offset=90
	.asciz	"int"                   ## string offset=99
	.asciz	"_type_anchor"          ## string offset=103
	.asciz	"U"                     ## string offset=116
	.asciz	"raw"                   ## string offset=118
	.asciz	"unsigned int"          ## string offset=122
	.asciz	"a"                     ## string offset=135
	.asciz	"b"                     ## string offset=137
	.asciz	"c"                     ## string offset=139
	.asciz	"d"                     ## string offset=141
	.asciz	"e"                     ## string offset=143
	.asciz	"f"                     ## string offset=145
	.asciz	"g"                     ## string offset=147
	.asciz	"main"                  ## string offset=149
	.asciz	"u"                     ## string offset=154
	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                       ## Abbreviation Code
	.byte	17                      ## DW_TAG_compile_unit
	.byte	1                       ## DW_CHILDREN_yes
	.byte	37                      ## DW_AT_producer
	.byte	14                      ## DW_FORM_strp
	.byte	19                      ## DW_AT_language
	.byte	5                       ## DW_FORM_data2
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	66                      ## DW_AT_stmt_list
	.byte	23                      ## DW_FORM_sec_offset
	.byte	27                      ## DW_AT_comp_dir
	.byte	14                      ## DW_FORM_strp
	.ascii	"\264B"                 ## DW_AT_GNU_pubnames
	.byte	25                      ## DW_FORM_flag_present
	.ascii	"\341\177"              ## DW_AT_APPLE_optimized
	.byte	25                      ## DW_FORM_flag_present
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	2                       ## Abbreviation Code
	.byte	52                      ## DW_TAG_variable
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	28                      ## DW_AT_const_value
	.byte	15                      ## DW_FORM_udata
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	3                       ## Abbreviation Code
	.byte	38                      ## DW_TAG_const_type
	.byte	0                       ## DW_CHILDREN_no
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	4                       ## Abbreviation Code
	.byte	36                      ## DW_TAG_base_type
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	62                      ## DW_AT_encoding
	.byte	11                      ## DW_FORM_data1
	.byte	11                      ## DW_AT_byte_size
	.byte	11                      ## DW_FORM_data1
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	5                       ## Abbreviation Code
	.byte	52                      ## DW_TAG_variable
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	2                       ## DW_AT_location
	.byte	24                      ## DW_FORM_exprloc
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	6                       ## Abbreviation Code
	.byte	22                      ## DW_TAG_typedef
	.byte	0                       ## DW_CHILDREN_no
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	7                       ## Abbreviation Code
	.byte	23                      ## DW_TAG_union_type
	.byte	1                       ## DW_CHILDREN_yes
	.byte	11                      ## DW_AT_byte_size
	.byte	11                      ## DW_FORM_data1
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	8                       ## Abbreviation Code
	.byte	13                      ## DW_TAG_member
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	56                      ## DW_AT_data_member_location
	.byte	11                      ## DW_FORM_data1
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	9                       ## Abbreviation Code
	.byte	13                      ## DW_TAG_member
	.byte	0                       ## DW_CHILDREN_no
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	56                      ## DW_AT_data_member_location
	.byte	11                      ## DW_FORM_data1
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	10                      ## Abbreviation Code
	.byte	19                      ## DW_TAG_structure_type
	.byte	1                       ## DW_CHILDREN_yes
	.byte	11                      ## DW_AT_byte_size
	.byte	11                      ## DW_FORM_data1
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	11                      ## Abbreviation Code
	.byte	13                      ## DW_TAG_member
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	13                      ## DW_AT_bit_size
	.byte	11                      ## DW_FORM_data1
	.byte	107                     ## DW_AT_data_bit_offset
	.byte	11                      ## DW_FORM_data1
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	12                      ## Abbreviation Code
	.byte	46                      ## DW_TAG_subprogram
	.byte	0                       ## DW_CHILDREN_no
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.byte	64                      ## DW_AT_frame_base
	.byte	24                      ## DW_FORM_exprloc
	.byte	122                     ## DW_AT_call_all_calls
	.byte	25                      ## DW_FORM_flag_present
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	63                      ## DW_AT_external
	.byte	25                      ## DW_FORM_flag_present
	.ascii	"\341\177"              ## DW_AT_APPLE_optimized
	.byte	25                      ## DW_FORM_flag_present
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	13                      ## Abbreviation Code
	.byte	46                      ## DW_TAG_subprogram
	.byte	1                       ## DW_CHILDREN_yes
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.byte	64                      ## DW_AT_frame_base
	.byte	24                      ## DW_FORM_exprloc
	.byte	122                     ## DW_AT_call_all_calls
	.byte	25                      ## DW_FORM_flag_present
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	63                      ## DW_AT_external
	.byte	25                      ## DW_FORM_flag_present
	.ascii	"\341\177"              ## DW_AT_APPLE_optimized
	.byte	25                      ## DW_FORM_flag_present
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	14                      ## Abbreviation Code
	.byte	52                      ## DW_TAG_variable
	.byte	0                       ## DW_CHILDREN_no
	.byte	28                      ## DW_AT_const_value
	.byte	15                      ## DW_FORM_udata
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	0                       ## EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
.set Lset0, Ldebug_info_end0-Ldebug_info_start0 ## Length of Unit
	.long	Lset0
Ldebug_info_start0:
	.short	4                       ## DWARF version number
.set Lset1, Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
	.long	Lset1
	.byte	8                       ## Address Size (in bytes)
	.byte	1                       ## Abbrev [1] 0xb:0x112 DW_TAG_compile_unit
	.long	0                       ## DW_AT_producer
	.short	12                      ## DW_AT_language
	.long	47                      ## DW_AT_name
	.long 0                       ## DW_AT_stmt_list
	.long	71                      ## DW_AT_comp_dir
                                        ## DW_AT_GNU_pubnames
                                        ## DW_AT_APPLE_optimized
	.quad	0            ## DW_AT_low_pc
	.long 0
	.byte	2                       ## Abbrev [2] 0x2a:0x10 DW_TAG_variable
	.long	90                      ## DW_AT_name
	.long	91                      ## DW_AT_type
	.byte	1                       ## DW_AT_decl_file
	.byte	16                      ## DW_AT_decl_line
	.ascii	"\201\202\220\245\006"  ## DW_AT_const_value
	.byte	3                       ## Abbrev [3] 0x3a:0x5 DW_TAG_const_type
	.long	63                      ## DW_AT_type
	.byte	4                       ## Abbrev [4] 0x3f:0x7 DW_TAG_base_type
	.long	99                      ## DW_AT_name
	.byte	5                       ## DW_AT_encoding
	.byte	4                       ## DW_AT_byte_size
	.byte	5                       ## Abbrev [5] 0x46:0x15 DW_TAG_variable
	.long	103                     ## DW_AT_name
	.long	91                      ## DW_AT_type
	.byte	1                       ## DW_AT_decl_file
	.byte	15                      ## DW_AT_decl_line
	.byte	9                       ## DW_AT_location
	.byte	3
	.quad	__type_anchor
	.byte	6                       ## Abbrev [6] 0x5b:0xb DW_TAG_typedef
	.long	102                     ## DW_AT_type
	.long	116                     ## DW_AT_name
	.byte	1                       ## DW_AT_decl_file
	.byte	13                      ## DW_AT_decl_line
	.byte	7                       ## Abbrev [7] 0x66:0x6c DW_TAG_union_type
	.byte	4                       ## DW_AT_byte_size
	.byte	1                       ## DW_AT_decl_file
	.byte	1                       ## DW_AT_decl_line
	.byte	8                       ## Abbrev [8] 0x6a:0xc DW_TAG_member
	.long	118                     ## DW_AT_name
	.long	210                     ## DW_AT_type
	.byte	1                       ## DW_AT_decl_file
	.byte	3                       ## DW_AT_decl_line
	.byte	0                       ## DW_AT_data_member_location
	.byte	9                       ## Abbrev [9] 0x76:0x8 DW_TAG_member
	.long	126                     ## DW_AT_type
	.byte	1                       ## DW_AT_decl_file
	.byte	4                       ## DW_AT_decl_line
	.byte	0                       ## DW_AT_data_member_location
	.byte	10                      ## Abbrev [10] 0x7e:0x53 DW_TAG_structure_type
	.byte	4                       ## DW_AT_byte_size
	.byte	1                       ## DW_AT_decl_file
	.byte	4                       ## DW_AT_decl_line
	.byte	11                      ## Abbrev [11] 0x82:0xd DW_TAG_member
	.long	135                     ## DW_AT_name
	.long	210                     ## DW_AT_type
	.byte	1                       ## DW_AT_decl_file
	.byte	6                       ## DW_AT_decl_line
	.byte	8                       ## DW_AT_bit_size
	.byte	0                       ## DW_AT_data_bit_offset
	.byte	11                      ## Abbrev [11] 0x8f:0xd DW_TAG_member
	.long	137                     ## DW_AT_name
	.long	210                     ## DW_AT_type
	.byte	1                       ## DW_AT_decl_file
	.byte	7                       ## DW_AT_decl_line
	.byte	8                       ## DW_AT_bit_size
	.byte	8                       ## DW_AT_data_bit_offset
	.byte	11                      ## Abbrev [11] 0x9c:0xd DW_TAG_member
	.long	139                     ## DW_AT_name
	.long	210                     ## DW_AT_type
	.byte	1                       ## DW_AT_decl_file
	.byte	8                       ## DW_AT_decl_line
	.byte	6                       ## DW_AT_bit_size
	.byte	16                      ## DW_AT_data_bit_offset
	.byte	11                      ## Abbrev [11] 0xa9:0xd DW_TAG_member
	.long	141                     ## DW_AT_name
	.long	210                     ## DW_AT_type
	.byte	1                       ## DW_AT_decl_file
	.byte	9                       ## DW_AT_decl_line
	.byte	2                       ## DW_AT_bit_size
	.byte	22                      ## DW_AT_data_bit_offset
	.byte	11                      ## Abbrev [11] 0xb6:0xd DW_TAG_member
	.long	143                     ## DW_AT_name
	.long	210                     ## DW_AT_type
	.byte	1                       ## DW_AT_decl_file
	.byte	10                      ## DW_AT_decl_line
	.byte	6                       ## DW_AT_bit_size
	.byte	24                      ## DW_AT_data_bit_offset
	.byte	11                      ## Abbrev [11] 0xc3:0xd DW_TAG_member
	.long	145                     ## DW_AT_name
	.long	210                     ## DW_AT_type
	.byte	1                       ## DW_AT_decl_file
	.byte	11                      ## DW_AT_decl_line
	.byte	2                       ## DW_AT_bit_size
	.byte	30                      ## DW_AT_data_bit_offset
	.byte	0                       ## End Of Children Mark
	.byte	0                       ## End Of Children Mark
	.byte	4                       ## Abbrev [4] 0xd2:0x7 DW_TAG_base_type
	.long	122                     ## DW_AT_name
	.byte	7                       ## DW_AT_encoding
	.byte	4                       ## DW_AT_byte_size
	.byte	14                      ## Abbrev [14] 0x10b:0x10 DW_TAG_variable
	.ascii	"\201\202\220\245\006"  ## DW_AT_const_value
	.long	154                     ## DW_AT_name
	.byte	1                       ## DW_AT_decl_file
	.byte	21                      ## DW_AT_decl_line
	.long	91                      ## DW_AT_type
	.byte	0                       ## End Of Children Mark
	.byte	0                       ## End Of Children Mark
Ldebug_info_end0:
