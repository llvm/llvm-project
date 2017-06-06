# Test object to verify that llvm-dwarfdump handles a degenerate string offsets
# section.
#
# To generate the test object:
# llvm-mc -triple x86_64-unknown-linux dwarfdump-str-offsets-invalid-5.s -filetype=obj \
#         -o dwarfdump-str-offsets-invalid-5.x86_64.o
# Every unit contributes to the string_offsets table.
        .section .debug_str_offsets,"",@progbits
# A degenerate section, not enough for a single entry.
        .byte 2
