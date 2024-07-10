##  Test  --skip-line-zero option.
##
##  This test uses handcrafted assembly to produce the following line table:
##  Address            Line   Column File   ISA Discriminator OpIndex Flags
##  ------------------ ------ ------ ------ --- ------------- ------- -------------
##  0x0000000000001710      1      0      1   0             0       0
##  0x0000000000001714      0      0      1   0             0       0
##  0x0000000000001719      1      0      1   0             0       0
##  0x000000000000171b      1      0      1   0             0       0  end_sequence
##  0x00000000000016c0      0      0      1   0             0       0
##  0x00000000000016cf      2      0      1   0             0       0
##  0x00000000000016d4      0      0      1   0             0       0
##  0x00000000000016da      0      0      1   0             0       0  end_sequence

# REQUIRES: x86-registered-target

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux --fdebug-prefix-map=%t="" %s -o %t.o

## Check that without '--skip-line-zero', line number zero is displayed for the line-table entry which has no source correspondence.
# RUN: llvm-symbolizer --obj=%t.o 0x16d4 | FileCheck --strict-whitespace --match-full-lines --check-prefix=APPROX-DISABLE %s

# APPROX-DISABLE:main
# APPROX-DISABLE-NEXT:main.c:0:0

## Check that with '--skip-line-zero', the last non-zero line in the current sequence is displayed.
## If it fails to find in the current sequence then return the orignal computed line-zero for the queried address.
# RUN: llvm-symbolizer --obj=%t.o --skip-line-zero 0x16c0 | FileCheck --strict-whitespace --match-full-lines --check-prefix=APPROX-FAIL-ACROSS-SEQ %s

# APPROX-FAIL-ACROSS-SEQ:main
# APPROX-FAIL-ACROSS-SEQ-NEXT:main.c:0:0

## Check that with '--skip-line-zero', the last non-zero line in the current sequence is displayed.
# RUN: llvm-symbolizer --obj=%t.o --skip-line-zero 0x1717 | FileCheck --strict-whitespace --match-full-lines --check-prefix=APPROX-WITHIN-SEQ %s

# APPROX-WITHIN-SEQ:foo
# APPROX-WITHIN-SEQ-NEXT:main.c:1:0 (approximate)

## Check to ensure that '--skip-line-zero' only affects addresses having line-zero when more than one address is specified.
# RUN: llvm-symbolizer --obj=%t.o --skip-line-zero 0x16d4 0x1719 | FileCheck --strict-whitespace --match-full-lines --check-prefixes=APPROX-ENABLE,NO-APPROX %s

# APPROX-ENABLE:main
# APPROX-ENABLE-NEXT:main.c:2:0 (approximate)
# NO-APPROX:foo
# NO-APPROX-NEXT:main.c:1:0

## Check to ensure that '--skip-line-zero' with '--verbose' enabled displays correct approximate flag in verbose ouptut.
# RUN: llvm-symbolizer --obj=%t.o --skip-line-zero --verbose 0x1717 | FileCheck --strict-whitespace --match-full-lines --check-prefix=APPROX-VERBOSE %s

# APPROX-VERBOSE:foo
# APPROX-VERBOSE-NEXT:  Filename: main.c
# APPROX-VERBOSE-NEXT:  Function start filename: main.c
# APPROX-VERBOSE-NEXT:  Function start line: 1
# APPROX-VERBOSE-NEXT:  Function start address: 0x1710
# APPROX-VERBOSE-NEXT:  Line: 1
# APPROX-VERBOSE-NEXT:  Column: 0
# APPROX-VERBOSE-NEXT:  Approximate: true

## Check to ensure that '--skip-line-zero' with '--output-style=JSON' displays correct approximate flag in JSON output.
# RUN: llvm-symbolizer --obj=%t.o --skip-line-zero --output-style=JSON 0x1717 | FileCheck --strict-whitespace --match-full-lines --check-prefix=APPROX-JSON %s

# APPROX-JSON:[{"Address":"0x1717","ModuleName":"{{.*}}{{[/|\]+}}test{{[/|\]+}}tools{{[/|\]+}}llvm-symbolizer{{[/|\]+}}Output{{[/|\]+}}skip-line-zero.s.tmp.o","Symbol":[{"Approximate":true,"Column":0,"Discriminator":0,"FileName":"main.c","FunctionName":"foo","Line":1,"StartAddress":"0x1710","StartFileName":"main.c","StartLine":1}]}]

## main.c
## __attribute__((section("def"))) int foo() { return 1234; }
## int main(void) { return foo(); }
##
## Generated using
## clang -S -gdwarf-4 --target=x86_64-pc-linux -fdebug-prefix-map=/tmp="" main.c -o main.s
##
## Sections belonging to code segment(.text) are removed. Sections related to debug information(other than .debug_line) are modified. Section .debug_line is handwritten.

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
	.byte	0                               # DW_CHILDREN_no
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
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
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x55 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	29                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.quad	0                               # DW_AT_low_pc
	.long	.Ldebug_ranges0                 # DW_AT_ranges
	.byte	2                               # Abbrev [2] 0x26:0x19 DW_TAG_subprogram
	.quad	0x1710                          # DW_AT_low_pc (.Lfunc_begin0)
	.long	0x171b-0x1710                   # DW_AT_high_pc (.Lfunc_end0-.Lfunc_begin0)
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string2                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	88                              # DW_AT_type
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x3f:0x19 DW_TAG_subprogram
	.quad	0x16c0                          # DW_AT_low_pc (.Lfunc_begin1)
	.long	0x16da-0x16c0                   # DW_AT_high_pc (.Lfunc_end1-.Lfunc_begin1)
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string4                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	88                              # DW_AT_type
                                        # DW_AT_external
	.byte	4                               # Abbrev [4] 0x58:0x7 DW_TAG_base_type
	.long	.Linfo_string3                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	0x1710                          #.Lfunc_begin0
	.quad	0x171b                          #.Lfunc_end0
	.quad	0x16c0                          #.Lfunc_begin1
	.quad	0x16da                          #.Lfunc_end1
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 19.0.0git (git@github.com:ampandey-1995/llvm-project.git 90cd5ed938a244de794a5ce45a44845d20cf91f4)" # string offset=0
.Linfo_string1:
	.asciz	"main.c"                        # string offset=113
.Linfo_string2:
	.asciz	"foo"                           # string offset=120
.Linfo_string3:
	.asciz	"int"                           # string offset=124
.Linfo_string4:
	.asciz	"main"                          # string offset=128
	.ident	"clang version 19.0.0git (git@github.com:ampandey-1995/llvm-project.git 90cd5ed938a244de794a5ce45a44845d20cf91f4)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym foo
	.section	.debug_line,"",@progbits
.Lline_table_start0:
	.long .Lunit_end - .Lunit_start     # unit length
.Lunit_start:
	.short 4   # version
	.long .Lprologue_end - .Lprologue_start # header length
.Lprologue_start:
	.byte 1                                      # minimum_instruction_length
	.byte 1                                      # maximum_operations_per_instruction
	.byte 0                                      # default_is_stmt
	.byte -5                                     # line_base
	.byte 14                                     # line_range
	.byte 13                                     # opcode_base
	.byte 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1     # arguments in standard opcodes
	.byte 0                                      # end of include directories
	.asciz "main.c"                              # filename
	.byte 0                                      # directory index
	.byte 0                                      # modification time
	.byte 0                                      # length of file (unavailable)
	.byte 0                                      # end of filenames
.Lprologue_end:
	.byte 0x00, 9, 2                             # DW_LNE_set_address
	.quad 0x1710                                 # Address Value
	.byte 0x01                                   # DW_LNS_copy
	.byte 0x49                                   # (address += 4,  line += -1,  op-index += 0)
	.byte 0x59                                   # (address += 5,  line += 1,  op-index += 0)
	.byte 0x02                                   # DW_LNS_advance_pc
	.uleb128 0x02                                # (addr += 2, op-index += 0)
	.byte 0x00, 1, 1                             # DW_LNE_end_sequence
	.byte 0x00, 9, 2                             # DW_LNE_set_address
	.quad 0x16c0                                 # Address Value
	.byte 0x11                                   # (address += 0,  line += -1,  op-index += 0)
	.byte 0xe6                                   # (address += 15,  line += 0,  op-index += 0)
	.byte 0x56                                   # (address += 5,  line += -2,  op-index += 0)
	.byte 0x02                                   # DW_LNS_advance_pc
	.uleb128 0x06                                # (addr += 6, op-index += 0)
	.byte 0x00, 1, 1                             # DW_LNE_end_sequence
.Lunit_end:
