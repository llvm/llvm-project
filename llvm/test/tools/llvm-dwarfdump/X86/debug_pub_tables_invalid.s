# RUN: llvm-mc -triple x86_64 %s -filetype=obj -o %t
# RUN: not llvm-dwarfdump -v %t 2>&1 | FileCheck %s

# CHECK:      .debug_pubnames contents:
# CHECK-NEXT: error: unexpected end of data at offset 0x1 while reading [0x0, 0x4)

# CHECK:      .debug_pubtypes contents:
# CHECK-NEXT: error: unexpected end of data at offset 0x1 while reading [0x0, 0x4)

# CHECK:      .debug_gnu_pubnames contents:
# CHECK-NEXT: error: unexpected end of data at offset 0x1 while reading [0x0, 0x4)

# CHECK:      .debug_gnu_pubtypes contents:
# CHECK-NEXT: error: unexpected end of data at offset 0x1 while reading [0x0, 0x4)

    .section .debug_pubnames,"",@progbits
    .byte 0

    .section .debug_pubtypes,"",@progbits
    .byte 0

    .section .debug_gnu_pubnames,"",@progbits
    .byte 0

    .section .debug_gnu_pubtypes,"",@progbits
    .byte 0
