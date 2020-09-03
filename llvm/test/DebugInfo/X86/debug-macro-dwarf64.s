## This checks that llvm-dwarfdump can parse and dump DWARF64 macro units.

# RUN: llvm-mc -triple x86_64 -filetype=obj %s -o - \
# RUN:   | llvm-dwarfdump -debug-macro - \
# RUN:   | FileCheck %s

# CHECK:      .debug_macro contents:

# CHECK:      0x00000000:
# CHECK-NEXT: macro header: version = 0x0005, flags = 0x03, format = DWARF64, debug_line_offset = 0x0000000000000000
# CHECK-NEXT: DW_MACRO_start_file - lineno: 0 filenum: 0
# CHECK-NEXT:   DW_MACRO_import - import offset: 0x00000000[[MACRO1OFF:[[:xdigit:]]{8}]]
# CHECK-NEXT: DW_MACRO_end_file

# CHECK:      0x[[MACRO1OFF]]:
# CHECK-NEXT: macro header: version = 0x0005, flags = 0x01, format = DWARF64
# CHECK-NEXT: DW_MACRO_define_strp - lineno: 1 macro: FOO 1
# CHECK-NEXT: DW_MACRO_undef_strp - lineno: 9 macro: BAR
# CHECK-NEXT: DW_MACRO_undef_strp - lineno: 15 macro: BAZ

    .section .debug_macro, "", @progbits
.LMacro0:
    .short 5            # Version
    .byte 3             # Flags: offset_size_flag | debug_line_offset_flag
    .quad 0             # Debug Line Offset
    .byte 3             # DW_MACRO_start_file
    .uleb128 0          # Line
    .uleb128 0          # File
    .byte 7             # DW_MACRO_import
    .quad .LMacro1
    .byte 4             # DW_MACRO_end_file
    .byte 0             # End macro unit
.LMacro1:
    .short 5            # Version
    .byte 1             # Flags: offset_size_flag
    .byte 5             # DW_MACRO_define_strp
    .uleb128 1          # Line
    .quad .LStr0        # "FOO 1"
    .byte 6             # DW_MACRO_undef_strp
    .uleb128 9          # Line
    .quad .LStr1        # "BAR"
    .byte 6             # DW_MACRO_undef_strp
    .uleb128 15         # Line
    .quad .LStr2        # "BAZ"
    .byte 0             # End macro unit

    .section .debug_str, "MS", @progbits, 1
.LStr0:
    .asciz "FOO 1"
.LStr1:
    .asciz "BAR"
.LStr2:
    .asciz "BAZ"
