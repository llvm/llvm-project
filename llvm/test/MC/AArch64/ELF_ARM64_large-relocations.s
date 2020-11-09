# RUN: llvm-mc -triple=arm64-none-linux-gnu -large-code-model -filetype=obj -o %T/large-reloc.o %s
# RUN: llvm-rtdyld -triple=arm64-none-linux-gnu -verify -map-section large-reloc.o,.eh_frame=0x10000 -map-section large-reloc.o,.text=0xffff000000000000 -check=%s %T/large-reloc.o
# RUN-BE: llvm-mc -triple=aarch64_be-none-linux-gnu -large-code-model -filetype=obj -o %T/be-large-reloc.o %s
# RUN-BE: llvm-rtdyld -triple=aarch64_be-none-linux-gnu -verify -map-section be-large-reloc.o,.eh_frame=0x10000 -map-section be-large-reloc.o,.text=0xffff000000000000 -check=%s %T/be-large-reloc.o

        .text
        .globl  g
        .p2align        2
        .type   g,@function
g:
        .cfi_startproc
        mov      x0, xzr
        ret
        .Lfunc_end0:
        .size   g, .Lfunc_end0-g
        .cfi_endproc

# Skip the CIE and load the 8 bytes PC begin pointer.
# Assuming the CIE and the FDE length are both 4 bytes.
# rtdyld-check: *{8}(section_addr(large-reloc.o, .eh_frame) + (*{4}(section_addr(large-reloc.o, .eh_frame))) + 0xc) = g - (section_addr(large-reloc.o, .eh_frame) + (*{4}(section_addr(large-reloc.o, .eh_frame))) + 0xc)
