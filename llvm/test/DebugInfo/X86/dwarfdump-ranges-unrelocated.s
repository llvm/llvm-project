# RUN: llvm-mc -triple x86_64-pc-linux -filetype=obj %s -o %t
# RUN: llvm-dwarfdump -v %t | FileCheck %s

# CHECK: .debug_info contents:
# CHECK: DW_TAG_compile_unit
# CHECK: DW_AT_ranges [DW_FORM_sec_offset] (0x00000000
# CHECK-NEXT:  [0x0000000000000000, 0x0000000000000001) ".text.foo1"
# CHECK-NEXT:  [0x0000000000000000, 0x0000000000000002) ".text.foo2" [4]
# CHECK-NEXT:  [0x0000000000000000, 0x0000000000000003) ".text.foo2" [5])

# CHECK: .debug_ranges contents:
# CHECK:   00000000 0000000000000000 0000000000000001
# CHECK:   00000000 0000000000000000 0000000000000002
# CHECK:   00000000 0000000000000000 0000000000000003
# CHECK:   00000000 <End of list>

# RUN: llvm-dwarfdump %t | FileCheck %s --check-prefix=BRIEF
# BRIEF: DW_TAG_compile_unit
# BRIEF: DW_AT_ranges         (0x00000000
# BRIEF-NEXT:  [0x0000000000000000, 0x0000000000000001)
# BRIEF-NEXT:  [0x0000000000000000, 0x0000000000000002)
# BRIEF-NEXT:  [0x0000000000000000, 0x0000000000000003))

# RUN: llvm-dwarfdump -diff %t | FileCheck %s --check-prefix=DIFF
# DIFF: DW_TAG_compile_unit
# DIFF-NEXT: DW_AT_producer	()
# DIFF-NEXT: DW_AT_language	(DW_LANG_C_plus_plus)
# DIFF-NEXT: DW_AT_name	()
# DIFF-NEXT: DW_AT_stmt_list	()
# DIFF-NEXT: DW_AT_comp_dir	()
# DIFF-NEXT: DW_AT_low_pc	()
# DIFF-NEXT: DW_AT_ranges	()

## Asm code for testcase is a reduced and modified output from next
## invocation and source:
# clang test.cpp -S -o test.s -gmlt -ffunction-sections
# test.cpp:
#   void foo1() { }
#   void foo2() { }

.section .text.foo1,"ax",@progbits
.Lfunc_begin0:
 nop
.Lfunc_end0:

.section .text.foo2,"ax",@progbits, unique, 1
.Lfunc_begin1:
 nop
 nop
.Lfunc_end1:

.section .text.foo2,"ax",@progbits, unique, 2
.Lfunc_begin2:
 nop
 nop
 nop
.Lfunc_end2:

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
.quad .Lfunc_begin2
.quad .Lfunc_end2
.quad 0
.quad 0
