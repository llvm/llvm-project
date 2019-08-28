# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj -o %t/macho_zero_fill_align.o %s
# RUN: llvm-jitlink -noexec %t/macho_zero_fill_align.o -entry higher_zero_fill_align

        .section        __DATA,__data
        .globl low_aligned_data
        .p2align  0
low_aligned_data:
        .byte 42

        .globl higher_zero_fill_align
.zerofill __DATA,__zero_fill,higher_zero_fill_align,8,3

.subsections_via_symbols
