# RUN: llvm-mc -triple=powerpc64-unknown-linux-gnu -filetype=obj %s | \
# RUN: llvm-readobj -r - | FileCheck %s

        .section .opd,"aw",@progbits
access_int64:
        .quad .L.access_int64
        .quad .TOC.@tocbase
        .quad 0
        .text
.L.access_int64:
        ld 4, .LC1@toc(2)
        bl sin

        .section .toc,"aw",@progbits
.LC1:
        .tc number64[TC],number64
        .data
        .globl number64
number64:
        .quad	10

# CHECK:      Relocations [

# The relocations in .rela.text are the 'number64' load using a
# R_PPC64_TOC16_DS against the .toc and the 'sin' external function
# address using a R_PPC64_REL24
# CHECK:        Section ({{[0-9]+}}) .rela.text {
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TOC16_DS .toc
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_REL24    sin
# CHECK-NEXT:   }

# The .opd entry for the 'access_int64' function creates 2 relocations:
# 1. A R_PPC64_ADDR64 against the .text segment plus addend (the function
#    address itself);
# 2. And a R_PPC64_TOC against no symbol (the linker will replace for the
#    module's TOC base).
# CHECK:        Section ({{[0-9]+}}) .rela.opd {
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_ADDR64 .text 0x0
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_TOC - 0x0

# Finally the TOC creates the relocation for the 'number64'.
# CHECK:        Section ({{[0-9]+}}) .rela.toc {
# CHECK-NEXT:     0x{{[0-9,A-F]+}} R_PPC64_ADDR64 number64 0x0
# CHECK-NEXT:   }

# CHECK-NEXT: ]
