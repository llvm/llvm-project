# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o
# RUN: llvm-dwarfdump -v %t.o | FileCheck --check-prefix=INVALIDSECTIONLENGTH %s
#
# Test object to verify that llvm-dwarfdump handles a degenerate string offsets
# section.
#
# Every unit contributes to the string_offsets table.
        .section .debug_str_offsets,"",@progbits
# A degenerate section, not enough for a single entry.
        .byte 2

# INVALIDSECTIONLENGTH: .debug_str_offsets contents:
# INVALIDSECTIONLENGTH: 0x00000000: Gap, length = 1
