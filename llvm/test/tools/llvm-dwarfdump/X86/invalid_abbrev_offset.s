# RUN: llvm-mc %s -filetype obj -triple x86_64 -o %t
# RUN: llvm-dwarfdump -debug-info -debug-types %t | FileCheck %s

# CHECK:      .debug_info contents:
# CHECK-NEXT: Compile Unit: {{.+}}, abbr_offset = 0x00a5 (invalid),
# CHECK-NEXT: <compile unit can't be parsed!>

# CHECK:      .debug_types contents:
# CHECK-NEXT: Type Unit: {{.+}}, abbr_offset = 0x00a5 (invalid), addr_size = 0x08, name = '',
# CHECK-NEXT: <type unit can't be parsed!>

    .section .debug_info,"",@progbits
    .long .LCUEnd-.LCUVersion   # Length of Unit
.LCUVersion:
    .short 4                    # DWARF version number
    .long 0xa5                  # Offset Into Abbrev. Section (invalid)
    .byte 8                     # Address Size
    .byte 1                     # Abbreviation code
.LCUEnd:

    .section .debug_types,"",@progbits
.LTUBegin:
    .long .LTUEnd-.LTUVersion   # Length of Unit
.LTUVersion:
    .short 4                    # DWARF version number
    .long 0xa5                  # Offset Into Abbrev. Section (invalid)
    .byte 8                     # Address Size
    .quad 0x0011223344556677    # Type Signature
    .long .LTUType-.LTUBegin    # Type offset
.LTUType:
    .byte 1                     # Abbreviation code
.LTUEnd:
