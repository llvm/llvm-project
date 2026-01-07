# REQUIRES: system-linux
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=systemz-unknown-linux -position-independent \
# RUN:     -mcpu=z16 -filetype=obj -o %t/elf_reloc.o %s

# RUN: llvm-jitlink -noexec \
# RUN:    -abs external_addr12=0xffe \
# RUN:    -abs external_addr16=0xfffe \
# RUN:    -abs external_addr24=0xffffe \
# RUN:     %t/elf_reloc.o -check %s


        .text
        .section        .text.main
        .globl  main
        .p2align        4
        .type   main,@function
main:
        br   %r14
        .size main, .-main

# R_390_PLT16DBL
# jitlink-check: *{2}(test_plt16dbl + 4) = \
# jitlink-check:     (stub_addr(elf_reloc.o, external_addr16) - \
# jitlink-check:                test_plt16dbl) >> 1
        .globl  test_plt16dbl
        .p2align 4 
test_plt16dbl:
	bpp   0, external_addr16@plt, 0
        .size test_plt16dbl,.-test_plt16dbl

# R_390_PLT12DBL
# jitlink-check: ((*{2}(test_plt12dbl + 1)) & 0x0fff) = \
# jitlink-check:      (stub_addr(elf_reloc.o, external_addr12) - \
# jitlink-check:                 test_plt12dbl) >> 1
        .globl  test_plt12dbl
        .p2align 4 
test_plt12dbl:
        bprp  0, external_addr12@plt, 0 
        .size test_plt12dbl,.-test_plt12dbl

# R_390_PLT24DBL
# jitlink-check: ((*{4}(test_plt24dbl + 2)) & 0x0ffffff) = \
# jitlink-check:       (stub_addr(elf_reloc.o, external_addr24) - \
# jitlink-check:                  test_plt24dbl) >> 1
        .globl  test_plt24dbl
        .p2align 4 
test_plt24dbl:
        bprp  0, 0, external_addr24@plt 
        .size test_plt24dbl,.-test_plt24dbl

