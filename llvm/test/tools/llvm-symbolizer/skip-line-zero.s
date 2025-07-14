##  Test the "--skip-line-zero" option.
##
##  This test uses handcrafted assembly to produce the following line table:
##  Address            Line   Column File   ISA Discriminator OpIndex Flags
##  ------------------ ------ ------ ------ --- ------------- ------- -------------
##  0x0000000000001710      1      0      1   0             0       0
##  0x0000000000001714      0      0      1   0             0       0
##  0x0000000000001719      1      2      1   0             0       0
##  0x000000000000171b      1      2      1   0             0       0  end_sequence
##  0x00000000000016c0      0      0      1   0             0       0
##  0x00000000000016cf      2      0      1   0             0       0
##  0x00000000000016d4      0      0      1   0             0       0
##  0x00000000000016d9      0      0      1   0             0       0
##  0x00000000000016df      0      0      1   0             0       0  end_sequence

# REQUIRES: x86-registered-target

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o

## Check that without '--skip-line-zero', line zero is displayed for a line-table entry which has no source correspondence.
# RUN: llvm-symbolizer --obj=%t.o -f=none 0x16d4 | FileCheck --strict-whitespace --match-full-lines --check-prefix=DISABLE %s

# DISABLE:main.c:0:0

## Check that the '--skip-line-zero' does not cross sequence boundaries.
## If it fails to find in the current sequence then line zero is returned for the queried address.
# RUN: llvm-symbolizer --obj=%t.o -f=none --skip-line-zero 0x16c0 | FileCheck --strict-whitespace --match-full-lines --check-prefix=FAIL-ACROSS-SEQ %s

# FAIL-ACROSS-SEQ:main.c:0:0

## Check that with '--skip-line-zero', the last non-zero line in the current sequence is displayed.
# RUN: llvm-symbolizer --obj=%t.o -f=none --skip-line-zero 0x1717 | FileCheck --strict-whitespace --match-full-lines --check-prefix=WITHIN-SEQ %s

# WITHIN-SEQ:main.c:1:0 (approximate)

## Check that with '--skip-line-zero', multiple line zero rows are skipped within the current sequence.
# RUN: llvm-symbolizer --obj=%t.o -f=none --skip-line-zero 0x16d9 | FileCheck --strict-whitespace --match-full-lines --check-prefix=MULTIPLE-ROWS %s

# MULTIPLE-ROWS:main.c:2:0 (approximate)

## Check that '--skip-line-zero' only affects the line zero addresses when more than one address is specified.
# RUN: llvm-symbolizer --obj=%t.o -f=none --skip-line-zero 0x16d4 0x1719 | FileCheck --strict-whitespace --match-full-lines --check-prefixes=ENABLE,NO-APPROX %s

# ENABLE:main.c:2:0 (approximate)
# NO-APPROX:main.c:1:2

## Check to ensure that '--skip-line-zero' with '--verbose' enabled displays approximate flag in verbose ouptut.
# RUN: llvm-symbolizer --obj=%t.o -f=none --skip-line-zero --verbose 0x1717 | FileCheck --strict-whitespace --match-full-lines --check-prefix=VERBOSE %s

# VERBOSE:  Filename: main.c
# VERBOSE-NEXT:  Line: 1
# VERBOSE-NEXT:  Column: 0
# VERBOSE-NEXT:  Approximate: true

## Check to ensure that '--skip-line-zero' with '--output-style=JSON' displays approximate flag in JSON output.
# RUN: llvm-symbolizer --obj=%t.o -f=none --skip-line-zero --output-style=JSON 0x1717 | FileCheck --strict-whitespace --match-full-lines --check-prefix=JSON %s

# JSON:[{"Address":"0x1717","ModuleName":"{{.*}}{{[/|\]+}}skip-line-zero.s{{.*}}","Symbol":[{"Approximate":true,"Column":0,"Discriminator":0,"FileName":"main.c","FunctionName":"","Line":1,"StartAddress":"","StartFileName":"","StartLine":0}]}]

## main.c
## __attribute__((section("def"))) int foo() { return 1234; }
## int main(void) { return foo()+5678; }
##
## Generated using
## clang -S -gdwarf-4 --target=x86_64-pc-linux -fdebug-prefix-map=/tmp="" main.c -o main.s
##
## Sections belonging to code segment(.text) are removed. Sections related to debug information(other than .debug_line) are modified. Section .debug_line is handwritten. Section .debug_str is deleted.

	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
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
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.quad	0                               # DW_AT_low_pc
	.long	.Ldebug_ranges0                 # DW_AT_ranges
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	0x1710                          #.Lfunc_begin0
	.quad	0x171b                          #.Lfunc_end0
	.quad	0x16c0                          #.Lfunc_begin1
	.quad	0x16df                          #.Lfunc_end1
	.quad	0
	.quad	0
	.section	.debug_line,"",@progbits
.Lline_table_start0:
	.long	.Lunit_end - .Lunit_start     # unit length
.Lunit_start:
	.short	4   # version
	.long	.Lprologue_end - .Lprologue_start # header length
.Lprologue_start:
	.byte	1                                      # minimum_instruction_length
	.byte	1                                      # maximum_operations_per_instruction
	.byte	0                                      # default_is_stmt
	.byte	-5                                     # line_base
	.byte	14                                     # line_range
	.byte	13                                     # opcode_base
	.byte	0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1     # arguments in standard opcodes
	.byte	0                                      # end of include directories
	.asciz	"main.c"                             # filename
	.byte	0                                      # directory index
	.byte	0                                      # modification time
	.byte	0                                      # length of file (unavailable)
	.byte	0                                      # end of filenames
.Lprologue_end:
	.byte	0x00, 9, 2                             # DW_LNE_set_address
	.quad	0x1710                                 # Address Value
	.byte	0x01                                   # DW_LNS_copy
	.byte	0x49                                   # (address += 4,  line += -1,  op-index += 0)
	.byte	0x05, 2                                # DW_LNS_set_column (2)
	.byte	0x59                                   # (address += 5,  line += 1,  op-index += 0)
	.byte	0x02                                   # DW_LNS_advance_pc
	.uleb128 0x02                                # (addr += 2, op-index += 0)
	.byte	0x00, 1, 1                             # DW_LNE_end_sequence
	.byte	0x00, 9, 2                             # DW_LNE_set_address
	.quad	0x16c0                                 # Address Value
	.byte	0x11                                   # (address += 0,  line += -1,  op-index += 0)
	.byte	0xe6                                   # (address += 15,  line += 2,  op-index += 0)
	.byte	0x56                                   # (address += 5,  line += -2,  op-index += 0)
	.byte	0x58                                   # (address += 5,  line += 0,  op-index += 0)
	.byte	0x02                                   # DW_LNS_advance_pc
	.uleb128 0x06                                # (addr += 6, op-index += 0)
	.byte	0x00, 1, 1                             # DW_LNE_end_sequence
.Lunit_end:

