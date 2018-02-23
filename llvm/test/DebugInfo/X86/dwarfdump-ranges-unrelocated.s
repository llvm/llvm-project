# RUN: llvm-mc -triple x86_64-pc-linux -filetype=obj %s -o %t
# RUN: llvm-dwarfdump %t | FileCheck %s

# CHECK: .debug_ranges contents:
# CHECK:   00000000 0000000000000000 0000000000000001
# CHECK:   00000000 0000000000000000 0000000000000002
# CHECK:   00000000 <End of list>

## Asm code for testcase is a reduced output from next invocation and source:
# clang test.cpp -S -o test.s -gmlt -ffunction-sections
# test.cpp:
#   void foo1() { }  
#   void foo2() { }  

.section .text.foo1,"ax",@progbits
.Lfunc_begin0:
 nop
.Lfunc_end0:

.section .text.foo2,"ax",@progbits
.Lfunc_begin1:
 nop
 nop
.Lfunc_end1:

.section .debug_abbrev,"",@progbits
.byte 1                       # Abbreviation Code
.byte 17                      # DW_TAG_compile_unit
.byte 0                       # DW_CHILDREN_no
.byte 37                      # DW_AT_producer
.byte 14                      # DW_FORM_strp
.byte 19                      # DW_AT_language
.byte 5                       # DW_FORM_data2
.byte 3                       # DW_AT_name
.byte 14                      # DW_FORM_strp
.byte 16                      # DW_AT_stmt_list
.byte 23                      # DW_FORM_sec_offset
.byte 27                      # DW_AT_comp_dir
.byte 14                      # DW_FORM_strp
.byte 17                      # DW_AT_low_pc
.byte 1                       # DW_FORM_addr
.byte 85                      # DW_AT_ranges
.byte 23                      # DW_FORM_sec_offset
.byte 0                       # EOM(1)
.byte 0                       # EOM(2)
.byte 0                       # EOM(3)

.section .debug_info,"",@progbits
.Lcu_begin0:
.long 38                      # Length of Unit
.short 4                      # DWARF version number
.long .debug_abbrev           # Offset Into Abbrev. Section
.byte 8                       # Address Size (in bytes)
.byte 1                       # Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
.long 0                       # DW_AT_producer
.short 4                      # DW_AT_language
.long 0                       # DW_AT_name
.long 0                       # DW_AT_stmt_list
.long 0                       # DW_AT_comp_dir
.quad 0                       # DW_AT_low_pc
.long .Ldebug_ranges0         # DW_AT_ranges

.section .debug_ranges,"",@progbits
.Ldebug_ranges0:
.quad .Lfunc_begin0
.quad .Lfunc_end0
.quad .Lfunc_begin1
.quad .Lfunc_end1
.quad 0
.quad 0
