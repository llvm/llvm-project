# Test basics of debug_line parsing. This test uses a linker script which
# ensures the code is placed at the end of a module to test the boundary
# condition when the final end-of-sequence line table entry points to an address
# that is outside the range of memory covered by the module.

# REQUIRES: lld

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj > %t.o
# RUN: ld.lld --script=%S/Inputs/debug-line-basic.script %t.o -o %t
# RUN: %lldb %t -o "image dump line-table -v a.c" -o exit | FileCheck %s

	.text
	.globl	_start
_start:
# CHECK: Line table for /tmp/a.c
	.file	1 "/tmp/b.c"
	.loc	1 0 0
        nop
# CHECK-NEXT: 0x0000000000201000: /tmp/b.c, is_start_of_statement = TRUE{{$}}
	.loc	1 1 0
        nop
# CHECK-NEXT: 0x0000000000201001: /tmp/b.c:1, is_start_of_statement = TRUE{{$}}
        .loc   1 1 1
        nop
# CHECK-NEXT: 0x0000000000201002: /tmp/b.c:1:1, is_start_of_statement = TRUE{{$}}
        .loc   1 2 0 is_stmt 0
        nop
# CHECK-NEXT: 0x0000000000201003: /tmp/b.c:2{{$}}
        .loc   1 2 0 is_stmt 0 basic_block
        nop
# CHECK-NEXT: 0x0000000000201004: /tmp/b.c:2, is_start_of_basic_block = TRUE{{$}}
        .loc   1 2 0 is_stmt 0 prologue_end
        nop
# CHECK-NEXT: 0x0000000000201005: /tmp/b.c:2, is_prologue_end = TRUE{{$}}
        .loc   1 2 0 is_stmt 0 epilogue_begin
        nop
# CHECK-NEXT: 0x0000000000201006: /tmp/b.c:2, is_epilogue_begin = TRUE{{$}}
	.file  2 "/tmp/c.c"
	.loc   2 1 0 is_stmt 0
        nop
# CHECK-NEXT: 0x0000000000201007: /tmp/c.c:1{{$}}
# CHECK-NEXT: 0x0000000000201008: /tmp/c.c:1, is_terminal_entry = TRUE{{$}}

	.section	.text.f,"ax",@progbits
f:
        .loc   1 3 0 is_stmt 0
        nop
# CHECK-NEXT: 0x0000000000201008: /tmp/b.c:3{{$}}
# CHECK-NEXT: 0x0000000000201009: /tmp/b.c:3, is_terminal_entry = TRUE{{$}}


	.section	.debug_str,"MS",@progbits,1
.Linfo_string1:
	.asciz	"a.c"
.Linfo_string2:
	.asciz	"/tmp"
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	16                      # DW_AT_stmt_list
	.byte	23                      # DW_FORM_sec_offset
	.byte	27                      # DW_AT_comp_dir
	.byte	14                      # DW_FORM_strp
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Lcu_end0-.Lcu_start0   # Length of Unit
.Lcu_start0:
	.short	4                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
	.short	12                      # DW_AT_language
	.long	.Linfo_string1          # DW_AT_name
	.long	.Lline_table_start0     # DW_AT_stmt_list
	.long	.Linfo_string2          # DW_AT_comp_dir
.Lcu_end0:
	.section	.debug_line,"",@progbits
.Lline_table_start0:
