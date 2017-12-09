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

# INVALIDSECTIONLENGTH:      .debug_str_offsets contents:
# INVALIDSECTIONLENGTH-NOT:  contents:
# INVALIDSECTIONLENGTH:      error: size of .debug_str_offsets is not a multiple of 4.
