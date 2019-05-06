# REQUIRES: x86-registered-target

# RUN: llvm-mc -filetype=obj -triple=x86_64 -dwarf-version=5 %s -o %t.o
# RUN: llvm-symbolizer --obj=%t.o --functions=none 0 1 | FileCheck %s

# CHECK: /home/0.cc:1:0
# CHECK: /tmp/1.cc:2:1

.Lfunc_begin0:
  .file 0 "/home" "0.cc"
  .file 1 "/tmp" "1.cc"
  .loc 0 1 0                 # 0.cc:1:0
  nop
  .loc 1 2 1                 # 1.cc:2:1
  nop
  .loc 0 1 1 prologue_end    # 0.cc:1:1
.Lfunc_end0:

.section .debug_abbrev,"",@progbits
  .byte 1     # Abbreviation Code
  .byte 17    # DW_TAG_compile_unit
  .byte 1     # DW_children_yes
  .byte 16    # DW_AT_stmt_list
  .byte 23    # DW_FROM_sec_offset
  .byte 17    # DW_AT_low_pc
  .byte 1     # DW_FORM_addr
  .byte 18    # DW_AT_high_pc
  .byte 6     # DW_FORM_data4
  .byte 0     # EOM(1)
  .byte 0     # EOM(2)

  .byte 2     # Abbreviation Code
  .byte 46    # DW_TAG_subprogram
  .byte 0     # DW_children_no
  .byte 3     # DW_AT_name
  .byte 8     # DW_FORM_string
  .byte 17    # DW_AT_low_pc
  .byte 1     # DW_FORM_addr
  .byte 18    # DW_AT_high_pc
  .byte 6     # DW_FORM_data4
  .byte 0     # EOM(1)
  .byte 0     # EOM(2)

  .byte 0     # EOM(3)

.section .debug_info,"",@progbits
.Lcu_begin0:
  .long .Lcu_end0-.Lcu_begin0-4 # Length of Unit
  .short 5                      # DWARF version number
  .byte 1                       # DWARF Unit Type
  .byte 8                       # Address Size (in bytes)
  .long .debug_abbrev           # Offset Into Abbrev.
  .byte 1                       # Abbrev [1]
  .long 0                       # DW_AT_stmt_list
  .quad 0                       # DW_AT_low_pc
  .long 2                       # DW_AT_high_pc
  .byte 2                       # Abbrev [2] DW_TAG_subprogram
  .asciz "0.c"                  # DW_AT_name
  .quad 0                       # DW_AT_low_pc
  .long 2                       # DW_AT_high_pc
  .byte 0
.Lcu_end0:
