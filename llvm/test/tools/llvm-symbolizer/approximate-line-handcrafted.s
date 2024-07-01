##  Test  --skip-line-zero option.
##
##  This test illustrates the usage of handcrafted assembly to produce the following line table.
##  Address            Line   Column File   ISA Discriminator OpIndex Flags
##  ------------------ ------ ------ ------ --- ------------- ------- -------------
##  0x00000000000030a0      1     80      1   0             0       0  is_stmt prologue_end
##  0x00000000000030a6      1     80      1   0             0       0  is_stmt end_sequence
##  0x0000000000001000      0     68      1   0             0       0  is_stmt prologue_end
##  0x0000000000001008      2     39      1   0             0       0
##  0x0000000000001010      3     68      1   0             0       0  is_stmt prologue_end
##  0x0000000000001012      0     68      1   0             0       0
##  0x0000000000001017      3     68      1   0             0       0
##  0x0000000000001019      3     39      1   0             0       0
##  0x0000000000001020      5     1       2   0             0       0  is_stmt prologue_end
##  0x0000000000001026      5     1       2   0             0       0  is_stmt end_sequence

# REQUIRES: x86-registered-target

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux --fdebug-prefix-map=%t="" %s -o %t.o

## Check that with '--skip-line-zero', the last non-zero line in the current sequence is displayed.
## If it fails to find in the current sequence then return the orignal computed line-zero for the queried address.
# RUN: llvm-symbolizer --obj=%t.o --skip-line-zero 0x1000 | FileCheck --strict-whitespace --match-full-lines --check-prefix=APPROX-FAIL-ACROSS-SEQ %s

# APPROX-FAIL-ACROSS-SEQ:add
# APPROX-FAIL-ACROSS-SEQ-NEXT:definitions.h:0:68

## Check that with '--skip-line-zero', the last non-zero line in the current sequence is displayed.
# RUN: llvm-symbolizer --obj=%t.o --skip-line-zero 0x1012 | FileCheck --strict-whitespace --match-full-lines --check-prefix=APPROX-WITHIN-SEQ %s

# APPROX-WITHIN-SEQ:sub
# APPROX-WITHIN-SEQ-NEXT:definitions.h:3:68 (approximate)

## Check to ensure that '--skip-line-zero' only affects addresses having line-zero when more than one address is specified.
# RUN: llvm-symbolizer --obj=%t.o --skip-line-zero 0x1012 0x1020 | FileCheck --strict-whitespace --match-full-lines --check-prefixes=APPROX-WITHIN-SEQ,NO-APPROX %s

# NO-APPROX:main
# NO-APPROX-NEXT:main.c:5:1

## Check to ensure that '--skip-line-zero' with '--verbose' enabled displays correct approximate flag in verbose ouptut.
# RUN: llvm-symbolizer --obj=%t.o --skip-line-zero --verbose 0x1012 | FileCheck --strict-whitespace --match-full-lines --check-prefix=APPROX-VERBOSE %s

# APPROX-VERBOSE:sub
# APPROX-VERBOSE-NEXT:  Filename: definitions.h
# APPROX-VERBOSE-NEXT:  Function start filename: definitions.h
# APPROX-VERBOSE-NEXT:  Function start line: 3
# APPROX-VERBOSE-NEXT:  Function start address: 0x1010
# APPROX-VERBOSE-NEXT:  Line: 3
# APPROX-VERBOSE-NEXT:  Column: 68
# APPROX-VERBOSE-NEXT:  Approximate: true

## Check to ensure that '--skip-line-zero' with '--output-style=JSON' displays correct approximate flag in JSON output.
# RUN: llvm-symbolizer --obj=%t.o --skip-line-zero --output-style=JSON 0x1012 | FileCheck --strict-whitespace --match-full-lines --check-prefix=APPROX-JSON %s

# APPROX-JSON:[{"Address":"0x1012","ModuleName":"{{.*}}{{[/|\]+}}test{{[/|\]+}}tools{{[/|\]+}}llvm-symbolizer{{[/|\]+}}Output{{[/|\]+}}approximate-line-handcrafted.s.tmp.o","Symbol":[{"Approximate":true,"Column":68,"Discriminator":0,"FileName":"definitions.h","FunctionName":"sub","Line":3,"StartAddress":"0x1010","StartFileName":"definitions.h","StartLine":3}]}]

#--- definitions.h
#__attribute__((section(".dummy_section"))) int dummy_function(){ return 1234; }
#extern inline int add(int x, int y) { return (dummy_function() + x + y); }
#extern inline int sub(int x, int y) { return (dummy_function() - x - y); }

#--- main.c
#include "definitions.h"
#int main(void) {
# int x = 111;
# int y = 322;
# return add(x,y)+sub(y,x);
#}

#--- gen
#clang -S -O3 -gdwarf-4 --target=x86_64-pc-linux -fdebug-prefix-map=/proc/self/cwd="" -fdebug-prefix-map=./="" main.c -o main.s

#sed -i '1,83d' main.s                             # Delete .text and .dummy_section
#sed -i '4c\	.quad 0x1000                            #.Lfunc_begin1 base address' main.s
#sed -i '5c\	.quad 0x1010                            #.Lfunc_begin2-.Lfunc_begin1' main.s
#sed -i '6c\	.quad 0x1012                            #.Ltmp2-.Lfunc_begin1' main.s
#sed -i '9c\	.quad 0x1012                            #.Ltmp2-.Lfunc_begin1' main.s
#sed -i '10c\	.quad 0x101a                              #.Lfunc_end2-.Lfunc_begin1' main.s
#sed -i '156c\	.quad 0x30a0                            #.Lfunc_begin0 DW_AT_low_pc' main.s
#sed -i '157c\	.long 0x30a6                            #.Lfunc_end0-.Lfunc_begin0(DW_AT_high_pc)' main.s
#sed -i '167c\	.quad 0x1000                            #.Lfunc_begin1(DW_AT_low_pc)' main.s
#sed -i '168c\	.long 0x1009                            #.Lfunc_end1-.Lfunc_begin1 DW_AT_high_pc' main.s
#sed -i '194c\	.quad 0x1010                            #.Lfunc_begin2  DW_AT_low_pc' main.s
#sed -i '195c\	.long 0x101a                            #.Lfunc_end2-.Lfunc_begin2  DW_AT_high_pc' main.s
#sed -i '220c\	.quad 0x1020                            #.Lfunc_begin3 DW_AT_low_pc' main.s
#sed -i '221c\	.long 0x1026                            #.Lfunc_end3-.Lfunc_begin3 DW_AT_high_pc' main.s
#sed -i '252c\	.quad 0x30a0                            #.Lfunc_begin0' main.s
#sed -i '253c\	.quad 0x30a6                            #.Lfunc_end0' main.s
#sed -i '254c\	.quad 0x1000                            #.Lfunc_begin1' main.s
#sed -i '255c\	.quad 0x30a6                            #.Lfunc_end3' main.s
#sed -i '$a\	.long .Lunit_end - .Lunit_start     # unit length\
#	.Lunit_start:\
#	.short 4     # version\
#	.long .Lprologue_end - .Lprologue_start # header length\
#.Lprologue_start:\
#	.byte 1                                      # minimum_instruction_length\
#	.byte 1                                      # maximum_operations_per_instruction\
#	.byte 1                                      # default_is_stmt\
#	.byte -5                                     # line_base\
#	.byte 14                                     # line_range\
#	.byte 13                                     # opcode_base\
#	.byte 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1     # arguments in standard opcodes\
#	.byte 0                                      # end of include directories\
#	.asciz "definitions.h"                       # filename\
#	.byte 0                                      # reference to dir0\
#	.byte 0                                      # modification time\
#	.byte 0                                      # length of file (unavailable)\
#	.asciz "main.c"                              # filename\
#	.byte 0                                      # reference to dir0\
#	.byte 0                                      # modification time\
#	.byte 0                                      # length of file (unavailable)\
#	.byte 0                                      # end of filenames\
#.Lprologue_end:\
#	.byte 0x05, 66                               # DW_LNS_set_column (80)\
#	.byte 0x0a                                   # DW_LNS_set_prologue_end\
#	.byte 0x00, 9, 2                             # DW_LNS_set_address\
#	.quad 0x30a0                                 # Address Value (0x00000000000030a0)\
#	.byte 0x01                                   # DW_LNS_copy\
#	.byte 0x02                                   # DW_LNS_advance_pc\
#	.uleb128 0x06                                # (addr +=6, op-index +=0)\
#	.byte 0, 1, 1                                # DW_LNE_end_sequence\
#	.byte 0x05, 68                               # DW_LNS_set_column (68)\
#	.byte 0x0a                                   # DW_LNS_set_prologue_end\
#	.byte 0x00, 9, 2                             # DW_LNE_set_address\
#	.quad 0x1000                                 # Address Value (0x0000000000001000)\
#	.byte 0x11                                   # (address += 0,  line += -1,  op-index += 0)\
#	.byte 0x05, 39                               # DW_LNS_set_column (39)\
#	.byte 0x06                                   # DW_LNS_negate_stmt\
#	.byte 0x84                                   # (address += 8,  line += 2,  op-index += 0)\
#	.byte 0x05, 68                               # DW_LNS_set_column (68)\
#	.byte 0x06                                   # DW_LNS_negate_stmt\
#	.byte 0x0a                                   # DW_LNS_set_prologue_end\
#	.byte 0x83                                   # (address += 8,  line += 1,  op-index += 0)\
#	.byte 0x06                                   # DW_LNS_negate_stmt\
#	.byte 0x2b                                   # (address += 2,  line += -3,  op-index += 0)\
#	.byte 0x5b                                   # (address += 5,  line += 3,  op-index += 0)\
#	.byte 0x05, 39                               # DW_LNS_set_column (39)\
#	.byte 0x2e                                   # (address += 2,  line += 0,  op-index += 0)\
#	.byte 0x04, 2                                # DW_LNS_set_file (2)\
#	.byte 0x05, 1                                # DW_LNS_set_column (1)\
#	.byte 0x06                                   # DW_LNS_negate_stmt\
#	.byte 0x0a                                   # DW_LNS_set_prologue_end\
#	.byte 0x76                                   # (address += 7,  line += 2,  op-index += 0)\
#	.byte 0x02                                   # DW_LNS_advance_pc\
#	.uleb128 0x06                                # (addr += 6, op-index += 0)\
#	.byte 0, 1, 1                                # DW_LNE_end_sequence\
#	.Lunit_end:' main.s

#sed -n p main.s

#--- main.s
	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
	.quad	-1
	.quad 0x1000                          #.Lfunc_begin1 base address
	.quad 0x1010                          #.Lfunc_begin2-.Lfunc_begin1
	.quad 0x1012                          #.Ltmp2-.Lfunc_begin1
	.short	1                               # Loc expr size
	.byte	85                              # super-register DW_OP_reg5
	.quad 0x1012                          #.Ltmp2-.Lfunc_begin1
	.quad 0x101a                          #.Lfunc_end2-.Lfunc_begin1
	.short	4                               # Loc expr size
	.byte	243                             # DW_OP_GNU_entry_value
	.byte	1                               # 1
	.byte	85                              # super-register DW_OP_reg5
	.byte	159                             # DW_OP_stack_value
	.quad	0
	.quad	0
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
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.ascii	"\227B"                       # DW_AT_GNU_all_call_sites
	.byte	25                              # DW_FORM_flag_present
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.ascii	"\227B"                       # DW_AT_GNU_all_call_sites
	.byte	25                              # DW_FORM_flag_present
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	39                              # DW_AT_prototyped
	.byte	25                              # DW_FORM_flag_present
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	23                              # DW_FORM_sec_offset
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	28                              # DW_AT_const_value
	.byte	13                              # DW_FORM_sdata
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
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
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                             # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0xda DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	29                            # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.quad	0                               # DW_AT_low_pc
	.long	.Ldebug_ranges0                 # DW_AT_ranges
	.byte	2                               # Abbrev [2] 0x26:0x19 DW_TAG_subprogram
	.quad 0x30a0                          #.Lfunc_begin0 DW_AT_low_pc
	.long 0x30a6                          #.Lfunc_end0-.Lfunc_begin0(DW_AT_high_pc)
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string2                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	221                             # DW_AT_type
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x3f:0x34 DW_TAG_subprogram
	.quad 0x1000                          #.Lfunc_begin1(DW_AT_low_pc)
	.long 0x1009                          #.Lfunc_end1-.Lfunc_begin1 DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string4                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	221                             # DW_AT_type
                                        # DW_AT_external
	.byte	4                               # Abbrev [4] 0x58:0xd DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.long	.Linfo_string7                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	221                             # DW_AT_type
	.byte	4                               # Abbrev [4] 0x65:0xd DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.long	.Linfo_string8                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	221                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0x73:0x36 DW_TAG_subprogram
	.quad 0x1010                          #.Lfunc_begin2  DW_AT_low_pc
	.long 0x101a                          #.Lfunc_end2-.Lfunc_begin2  DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string5                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	221                             # DW_AT_type
                                        # DW_AT_external
	.byte	5                               # Abbrev [5] 0x8c:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc0                    # DW_AT_location
	.long	.Linfo_string7                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.long	221                             # DW_AT_type
	.byte	4                               # Abbrev [4] 0x9b:0xd DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.long	.Linfo_string8                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.long	221                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0xa9:0x34 DW_TAG_subprogram
	.quad 0x1020                          #.Lfunc_begin3 DW_AT_low_pc
	.long 0x1026                          #.Lfunc_end3-.Lfunc_begin3 DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string6                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	221                             # DW_AT_type
                                        # DW_AT_external
	.byte	6                               # Abbrev [6] 0xc2:0xd DW_TAG_variable
	.asciz	"\357"                          # DW_AT_const_value
	.long	.Linfo_string7                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.long	221                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0xcf:0xd DW_TAG_variable
	.ascii	"\302\002"                      # DW_AT_const_value
	.long	.Linfo_string8                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.long	221                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	7                               # Abbrev [7] 0xdd:0x7 DW_TAG_base_type
	.long	.Linfo_string3                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad 0x30a0                          #.Lfunc_begin0
	.quad 0x30a6                          #.Lfunc_end0
	.quad 0x1000                          #.Lfunc_begin1
	.quad 0x30a6                          #.Lfunc_end3
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.byte	0                               # string offset=0
.Linfo_string1:
	.asciz	"main.c"                        # string offset=1
.Linfo_string2:
	.asciz	"dummy_function"                # string offset=8
.Linfo_string3:
	.asciz	"int"                           # string offset=23
.Linfo_string4:
	.asciz	"add"                           # string offset=27
.Linfo_string5:
	.asciz	"sub"                           # string offset=31
.Linfo_string6:
	.asciz	"main"                          # string offset=35
.Linfo_string7:
	.asciz	"x"                             # string offset=40
.Linfo_string8:
	.asciz	"y"                             # string offset=42
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
	.long .Lunit_end - .Lunit_start     # unit length
	.Lunit_start:
	.short 4     # version
	.long .Lprologue_end - .Lprologue_start # header length
.Lprologue_start:
	.byte 1                                      # minimum_instruction_length
	.byte 1                                      # maximum_operations_per_instruction
	.byte 1                                      # default_is_stmt
	.byte -5                                     # line_base
	.byte 14                                     # line_range
	.byte 13                                     # opcode_base
	.byte 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1     # arguments in standard opcodes
	.byte 0                                      # end of include directories
	.asciz "definitions.h"                       # filename
	.byte 0                                      # reference to dir0
	.byte 0                                      # modification time
	.byte 0                                      # length of file (unavailable)
	.asciz "main.c"                              # filename
	.byte 0                                      # reference to dir0
	.byte 0                                      # modification time
	.byte 0                                      # length of file (unavailable)
	.byte 0                                      # end of filenames
.Lprologue_end:
	.byte 0x05, 66                               # DW_LNS_set_column (80)
	.byte 0x0a                                   # DW_LNS_set_prologue_end
	.byte 0x00, 9, 2                             # DW_LNS_set_address
	.quad 0x30a0                                 # Address Value (0x00000000000030a0)
	.byte 0x01                                   # DW_LNS_copy
	.byte 0x02                                   # DW_LNS_advance_pc
	.uleb128 0x06                                # (addr +=6, op-index +=0)
	.byte 0, 1, 1                                # DW_LNE_end_sequence
	.byte 0x05, 68                               # DW_LNS_set_column (68)
	.byte 0x0a                                   # DW_LNS_set_prologue_end
	.byte 0x00, 9, 2                             # DW_LNE_set_address
	.quad 0x1000                                 # Address Value (0x0000000000001000)
	.byte 0x11                                   # (address += 0,  line += -1,  op-index += 0)
	.byte 0x05, 39                               # DW_LNS_set_column (39)
	.byte 0x06                                   # DW_LNS_negate_stmt
	.byte 0x84                                   # (address += 8,  line += 2,  op-index += 0)
	.byte 0x05, 68                               # DW_LNS_set_column (68)
	.byte 0x06                                   # DW_LNS_negate_stmt
	.byte 0x0a                                   # DW_LNS_set_prologue_end
	.byte 0x83                                   # (address += 8,  line += 1,  op-index += 0)
	.byte 0x06                                   # DW_LNS_negate_stmt
	.byte 0x2b                                   # (address += 2,  line += -3,  op-index += 0)
	.byte 0x5b                                   # (address += 5,  line += 3,  op-index += 0)
	.byte 0x05, 39                               # DW_LNS_set_column (39)
	.byte 0x2e                                   # (address += 2,  line += 0,  op-index += 0)
	.byte 0x04, 2                                # DW_LNS_set_file (2)
	.byte 0x05, 1                                # DW_LNS_set_column (1)
	.byte 0x06                                   # DW_LNS_negate_stmt
	.byte 0x0a                                   # DW_LNS_set_prologue_end
	.byte 0x76                                   # (address += 7,  line += 2,  op-index += 0)
	.byte 0x02                                   # DW_LNS_advance_pc
	.uleb128 0x06                                # (addr += 6, op-index += 0)
	.byte 0, 1, 1                                # DW_LNE_end_sequence
	.Lunit_end:
