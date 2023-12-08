# RUN: llvm-mc -filetype=obj -o %t -triple x86_64-apple-macosx10.15.0 %s
# RUN: %lldb %t -o "target variable ug" -b \
# RUN:  | FileCheck --check-prefix=TARGET-VARIABLE %s
# RUN: %lldb %t -o "image lookup --verbose --address 0x0" -b \
# RUN:  | FileCheck --check-prefix=IMAGE-LOOKUP %s

# TARGET-VARIABLE: (lldb) target variable ug
# TARGET-VARIABLE: (U) ug = {
# TARGET-VARIABLE:   raw = 0
# TARGET-VARIABLE:    = (a = 0, b = 0, c = 0, d = 0, e = 0, f = 0)
# TARGET-VARIABLE: }

# IMAGE-LOOKUP: Summary:
# IMAGE-LOOKUP: Module: file =

# We are testing how DWARFExpression::Evaluate(...) in the case of
# DW_OP_deref_size deals with static variable.
#
# The address will be a ValueType::FileAddress type which we will not be able
# to turn into a load address but it is a section offset. We should be able to
# use Target::ReadMemory(...) to read the data in this case.
# Compiling at -O1 allows us to capture this case and test it.
#
# typedef union {
#   unsigned raw;
#   struct {
#      unsigned a : 8;
#      unsigned b : 8;
#      unsigned c : 6;
#      unsigned d : 2;
#      unsigned e : 6;
#      unsigned f : 2;
#   };
# } U;
#
# // This appears first in the debug info and pulls the type definition in...
# static U __attribute__((used)) _type_anchor;
# // ... then our useful variable appears last in the debug info and we can
# // tweak the assembly without needing to edit a lot of offsets by hand.
# static U ug;
#
# extern void f(U);
#
# // Omit debug info for main.
# __attribute__((nodebug))
# int main() {
#   ug.raw = 0x64A40101;
#   f(ug);
#   f((U)ug.raw);
# }
#
# Compiled as follows:
#
#   clang -O1 -gdwarf-4 dw_op_deref_size_test.c -S -o dw_op_deref_size_test.s
#
# Hand modified .s file to remoce various section that are not needed for this
# test

.zerofill __DATA,__bss,_ug.0,1,2        ## @ug.0
.zerofill __DATA,__bss,__type_anchor,4,2 ## @_type_anchor
	.no_dead_strip	__type_anchor
	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                               ## Abbreviation Code
	.byte	17                              ## DW_TAG_compile_unit
	.byte	1                               ## DW_CHILDREN_yes
	.byte	37                              ## DW_AT_producer
	.byte	14                              ## DW_FORM_strp
	.byte	19                              ## DW_AT_language
	.byte	5                               ## DW_FORM_data2
	.byte	3                               ## DW_AT_name
	.byte	14                              ## DW_FORM_strp
	.ascii	"\202|"                         ## DW_AT_LLVM_sysroot
	.byte	14                              ## DW_FORM_strp
	.ascii	"\357\177"                      ## DW_AT_APPLE_sdk
	.byte	14                              ## DW_FORM_strp
	.byte	16                              ## DW_AT_stmt_list
	.byte	23                              ## DW_FORM_sec_offset
	.byte	27                              ## DW_AT_comp_dir
	.byte	14                              ## DW_FORM_strp
	.ascii	"\341\177"                      ## DW_AT_APPLE_optimized
	.byte	25                              ## DW_FORM_flag_present
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	2                               ## Abbreviation Code
	.byte	52                              ## DW_TAG_variable
	.byte	0                               ## DW_CHILDREN_no
	.byte	3                               ## DW_AT_name
	.byte	14                              ## DW_FORM_strp
	.byte	73                              ## DW_AT_type
	.byte	19                              ## DW_FORM_ref4
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	2                               ## DW_AT_location
	.byte	24                              ## DW_FORM_exprloc
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	3                               ## Abbreviation Code
	.byte	22                              ## DW_TAG_typedef
	.byte	0                               ## DW_CHILDREN_no
	.byte	73                              ## DW_AT_type
	.byte	19                              ## DW_FORM_ref4
	.byte	3                               ## DW_AT_name
	.byte	14                              ## DW_FORM_strp
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	4                               ## Abbreviation Code
	.byte	23                              ## DW_TAG_union_type
	.byte	1                               ## DW_CHILDREN_yes
	.byte	11                              ## DW_AT_byte_size
	.byte	11                              ## DW_FORM_data1
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	5                               ## Abbreviation Code
	.byte	13                              ## DW_TAG_member
	.byte	0                               ## DW_CHILDREN_no
	.byte	3                               ## DW_AT_name
	.byte	14                              ## DW_FORM_strp
	.byte	73                              ## DW_AT_type
	.byte	19                              ## DW_FORM_ref4
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	56                              ## DW_AT_data_member_location
	.byte	11                              ## DW_FORM_data1
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	6                               ## Abbreviation Code
	.byte	13                              ## DW_TAG_member
	.byte	0                               ## DW_CHILDREN_no
	.byte	73                              ## DW_AT_type
	.byte	19                              ## DW_FORM_ref4
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	56                              ## DW_AT_data_member_location
	.byte	11                              ## DW_FORM_data1
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	7                               ## Abbreviation Code
	.byte	19                              ## DW_TAG_structure_type
	.byte	1                               ## DW_CHILDREN_yes
	.byte	11                              ## DW_AT_byte_size
	.byte	11                              ## DW_FORM_data1
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	8                               ## Abbreviation Code
	.byte	13                              ## DW_TAG_member
	.byte	0                               ## DW_CHILDREN_no
	.byte	3                               ## DW_AT_name
	.byte	14                              ## DW_FORM_strp
	.byte	73                              ## DW_AT_type
	.byte	19                              ## DW_FORM_ref4
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	13                              ## DW_AT_bit_size
	.byte	11                              ## DW_FORM_data1
	.byte	107                             ## DW_AT_data_bit_offset
	.byte	11                              ## DW_FORM_data1
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	9                               ## Abbreviation Code
	.byte	36                              ## DW_TAG_base_type
	.byte	0                               ## DW_CHILDREN_no
	.byte	3                               ## DW_AT_name
	.byte	14                              ## DW_FORM_strp
	.byte	62                              ## DW_AT_encoding
	.byte	11                              ## DW_FORM_data1
	.byte	11                              ## DW_AT_byte_size
	.byte	11                              ## DW_FORM_data1
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	0                               ## EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
.set Lset0, Ldebug_info_end0-Ldebug_info_start0 ## Length of Unit
	.long	Lset0
Ldebug_info_start0:
	.short	4                               ## DWARF version number
.set Lset1, Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
	.long	Lset1
	.byte	8                               ## Address Size (in bytes)
	.byte	1                               ## Abbrev [1] 0xb:0xd0 DW_TAG_compile_unit
	.long	0                               ## DW_AT_producer
	.short	12                              ## DW_AT_language
	.long	105                             ## DW_AT_name
	.long	129                             ## DW_AT_LLVM_sysroot
	.long	185                             ## DW_AT_APPLE_sdk
	.long 0
	.long	200                             ## DW_AT_comp_dir
                                        ## DW_AT_APPLE_optimized
	.byte	2                               ## Abbrev [2] 0x26:0x15 DW_TAG_variable
	.long	219                             ## DW_AT_name
	.long	59                              ## DW_AT_type
	.byte	1                               ## DW_AT_decl_file
	.byte	14                              ## DW_AT_decl_line
	.byte	9                               ## DW_AT_location
	.byte	3
	.quad	__type_anchor
	.byte	3                               ## Abbrev [3] 0x3b:0xb DW_TAG_typedef
	.long	70                              ## DW_AT_type
	.long	232                             ## DW_AT_name
	.byte	1                               ## DW_AT_decl_file
	.byte	11                              ## DW_AT_decl_line
	.byte	4                               ## Abbrev [4] 0x46:0x6c DW_TAG_union_type
	.byte	4                               ## DW_AT_byte_size
	.byte	1                               ## DW_AT_decl_file
	.byte	1                               ## DW_AT_decl_line
	.byte	5                               ## Abbrev [5] 0x4a:0xc DW_TAG_member
	.long	234                             ## DW_AT_name
	.long	178                             ## DW_AT_type
	.byte	1                               ## DW_AT_decl_file
	.byte	2                               ## DW_AT_decl_line
	.byte	0                               ## DW_AT_data_member_location
	.byte	6                               ## Abbrev [6] 0x56:0x8 DW_TAG_member
	.long	94                              ## DW_AT_type
	.byte	1                               ## DW_AT_decl_file
	.byte	3                               ## DW_AT_decl_line
	.byte	0                               ## DW_AT_data_member_location
	.byte	7                               ## Abbrev [7] 0x5e:0x53 DW_TAG_structure_type
	.byte	4                               ## DW_AT_byte_size
	.byte	1                               ## DW_AT_decl_file
	.byte	3                               ## DW_AT_decl_line
	.byte	8                               ## Abbrev [8] 0x62:0xd DW_TAG_member
	.long	251                             ## DW_AT_name
	.long	178                             ## DW_AT_type
	.byte	1                               ## DW_AT_decl_file
	.byte	4                               ## DW_AT_decl_line
	.byte	8                               ## DW_AT_bit_size
	.byte	0                               ## DW_AT_data_bit_offset
	.byte	8                               ## Abbrev [8] 0x6f:0xd DW_TAG_member
	.long	253                             ## DW_AT_name
	.long	178                             ## DW_AT_type
	.byte	1                               ## DW_AT_decl_file
	.byte	5                               ## DW_AT_decl_line
	.byte	8                               ## DW_AT_bit_size
	.byte	8                               ## DW_AT_data_bit_offset
	.byte	8                               ## Abbrev [8] 0x7c:0xd DW_TAG_member
	.long	255                             ## DW_AT_name
	.long	178                             ## DW_AT_type
	.byte	1                               ## DW_AT_decl_file
	.byte	6                               ## DW_AT_decl_line
	.byte	6                               ## DW_AT_bit_size
	.byte	16                              ## DW_AT_data_bit_offset
	.byte	8                               ## Abbrev [8] 0x89:0xd DW_TAG_member
	.long	257                             ## DW_AT_name
	.long	178                             ## DW_AT_type
	.byte	1                               ## DW_AT_decl_file
	.byte	7                               ## DW_AT_decl_line
	.byte	2                               ## DW_AT_bit_size
	.byte	22                              ## DW_AT_data_bit_offset
	.byte	8                               ## Abbrev [8] 0x96:0xd DW_TAG_member
	.long	259                             ## DW_AT_name
	.long	178                             ## DW_AT_type
	.byte	1                               ## DW_AT_decl_file
	.byte	8                               ## DW_AT_decl_line
	.byte	6                               ## DW_AT_bit_size
	.byte	24                              ## DW_AT_data_bit_offset
	.byte	8                               ## Abbrev [8] 0xa3:0xd DW_TAG_member
	.long	261                             ## DW_AT_name
	.long	178                             ## DW_AT_type
	.byte	1                               ## DW_AT_decl_file
	.byte	9                               ## DW_AT_decl_line
	.byte	2                               ## DW_AT_bit_size
	.byte	30                              ## DW_AT_data_bit_offset
	.byte	0                               ## End Of Children Mark
	.byte	0                               ## End Of Children Mark
	.byte	9                               ## Abbrev [9] 0xb2:0x7 DW_TAG_base_type
	.long	238                             ## DW_AT_name
	.byte	7                               ## DW_AT_encoding
	.byte	4                               ## DW_AT_byte_size
	.byte	2                               ## Abbrev [2] 0xb9:0x21 DW_TAG_variable
	.long	263                             ## DW_AT_name
	.long	59                              ## DW_AT_type
	.byte	1                               ## DW_AT_decl_file
	.byte	17                              ## DW_AT_decl_line
	.byte	21                              ## DW_AT_location
	.byte	3
	.quad	_ug.0
	.byte	148
	.byte	1
	.byte	16
	.ascii	"\201\202\220\245\006"
	.byte	30
	.byte	48
	.byte	34
	.byte	159
	.byte	0                               ## End Of Children Mark
Ldebug_info_end0:
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
  .zero 219
	.asciz	"_type_anchor"                  ## string offset=219
	.asciz	"U"                             ## string offset=232
	.asciz	"raw"                           ## string offset=234
	.asciz	"unsigned int"                  ## string offset=238
	.asciz	"a"                             ## string offset=251
	.asciz	"b"                             ## string offset=253
	.asciz	"c"                             ## string offset=255
	.asciz	"d"                             ## string offset=257
	.asciz	"e"                             ## string offset=259
	.asciz	"f"                             ## string offset=261
	.asciz	"ug"                            ## string offset=263
