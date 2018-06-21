# RUN: llvm-mc -triple x86_64-pc-linux -filetype=obj %s -o %t
# RUN: llvm-dwarfdump -v %t 2>%t.err | FileCheck %s
# RUN: FileCheck %s <%t.err -check-prefix=ERR

# CHECK: .debug_info contents:
# CHECK: 0x0000000b: DW_TAG_compile_unit [1]
# CHECK:             DW_AT_low_pc [DW_FORM_addr]       (0x0000000000000000)
# CHECK-NEXT:        DW_AT_ranges [DW_FORM_sec_offset] (0x00000000
# CHECK-NEXT:    [0x0000000000000000, 0x0000000000000001) ".text"
# CHECK-NEXT:    [0x0000000000000003, 0x0000000000000006) ".text"
# CHECK-NEXT:    [0x0000000000000001, 0x0000000000000002) ".text.foo1")

.text
.globl foo
.type foo,@function
foo:
.Lfunc_begin0:
  nop
.Ltmp0:
  nop
  nop
.Ltmp1:
  nop
  nop
  nop
.Ltmp2:

.section .text.foo1,"ax",@progbits
.Ltmp3:
 nop
.Ltmp4:
 nop
.Ltmp5:

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
.quad .Lfunc_begin0           # DW_AT_low_pc
.long .Ldebug_ranges0         # DW_AT_ranges

# A CU with an invalid DW_AT_ranges attribute
.Lcu_begin1:
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
.quad .Lfunc_begin0           # DW_AT_low_pc
.long 0x4000                  # DW_AT_ranges

# ERR: error: decoding address ranges: invalid range list offset 0x4000

# A CU where the DW_AT_ranges attribute points to an invalid range list.
.Lcu_begin2:
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
.quad .Lfunc_begin0           # DW_AT_low_pc
.long .Ldebug_ranges1         # DW_AT_ranges

.section .debug_ranges,"",@progbits
.Ldebug_ranges0:
 .quad .Lfunc_begin0-.Lfunc_begin0
 .quad .Ltmp0-.Lfunc_begin0
 .quad .Ltmp1-.Lfunc_begin0
 .quad .Ltmp2-.Lfunc_begin0
 .quad 0xFFFFFFFFFFFFFFFF
 .quad .text.foo1
 .quad .Ltmp4-.text.foo1
 .quad .Ltmp5-.text.foo1
 .quad 0
 .quad 0
.Ldebug_ranges1:
 .quad 0

# ERR: error: decoding address ranges: invalid range list entry at offset 0x50
