# Test that lldb can read a line table for an architecture with a different
# address size than the one that of the host.

# REQUIRES: lld, x86

# RUN: llvm-mc -triple i686-windows-gnu %s -filetype=obj > %t.o
# RUN: lld-link %t.o -out:%t.exe -debug:dwarf -entry:entry -subsystem:console -lldmingw
# RUN: %lldb %t.exe -o "image dump line-table -v win-i386-line-table.c" -b | FileCheck %s

# CHECK: Line table for win-i386-line-table.c in `win-i386-line-table.s.tmp.exe
# CHECK: 0x00401000: win-i386-line-table.c:2:1
# CHECK: 0x00401001: win-i386-line-table.c:2:1

        .text
        .file   "win-i386-line-table.c"
        .globl  _entry                  # -- Begin function entry
_entry:                                 # @entry
        .file   1 "" "win-i386-line-table.c"
        .loc    1 1 0                   # win-i386-line-table.c:1:0
        .cfi_sections .debug_frame
        .cfi_startproc
        .loc    1 2 1 prologue_end      # win-i386-line-table.c:2:1
        retl
        .cfi_endproc
                                        # -- End function
        .section        .debug_str,"dr"
Linfo_string1:
        .asciz  "win-i386-line-table.c"
        .section        .debug_abbrev,"dr"
Lsection_abbrev:
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   16                      # DW_AT_stmt_list
        .byte   23                      # DW_FORM_sec_offset
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)
        .section        .debug_info,"dr"
Lsection_info:
Lcu_begin0:
        .long   Ldebug_info_end0-Ldebug_info_start0 # Length of Unit
Ldebug_info_start0:
        .short  4                       # DWARF version number
        .secrel32       Lsection_abbrev # Offset Into Abbrev. Section
        .byte   4                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] 0xb:0x2d DW_TAG_compile_unit
        .secrel32       Linfo_string1   # DW_AT_name
        .secrel32       Lline_table_start0 # DW_AT_stmt_list
        .byte   0                       # End Of Children Mark
Ldebug_info_end0:
        .section        .debug_line,"dr"
Lline_table_start0:
