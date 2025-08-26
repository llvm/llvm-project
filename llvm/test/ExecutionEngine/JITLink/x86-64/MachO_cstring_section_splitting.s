# REQUIRES: asserts
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj -o %t %s
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec \
# RUN:              -entry hook %t 2>&1 \
# RUN:              | FileCheck %s
#
# Verify that we split C string literals on null-terminators, rather than on
# symbol boundaries. We expect four dead-stripped symbols: l_str.0, l_str.2,
# L_str.3, l_str.4, and the auto-generated symbol for the start of the "defghi"
# string. We also verify that there are only two dead-stripped blocks, since
# l_str.3 should not have split the block started at "def"... (since this is a
# C string section we should be splitting on null characters instead of
# symbols).
#
# CHECK:      Dead-stripping defined symbols:
# CHECK-NEXT: linkage: strong, scope: local, dead
# CHECK-NEXT: linkage: strong, scope: local, dead
# CHECK-NEXT: linkage: strong, scope: local, dead
# CHECK-NEXT: linkage: strong, scope: local, dead
# CHECK-NEXT: Dead-stripping blocks:
# CHECK-NEXT: content, align = 1, align-ofs = 0, section = __TEXT,__cstring
# CHECK-NEXT: content, align = 1, align-ofs = 0, section = __TEXT,__cstring
# CHECK-NEXT: Removing unused external symbols

        .section        __DATA,__data
        .globl  hook
        .p2align        2
hook:
        .quad   l_str.1

        .section        __TEXT,__cstring,cstring_literals
l_str.0:
l_str.1:
        .asciz  "abc"
l_str.2:
        .asciz  ""
# anonymous start for "defghi", split in the middle by l_str.3. We expect this
# to be dead-stripped as a single block.
        .byte   'd'
        .byte   'e'
        .byte   'f'
l_str.3:
        .asciz  "ghi"

.subsections_via_symbols
