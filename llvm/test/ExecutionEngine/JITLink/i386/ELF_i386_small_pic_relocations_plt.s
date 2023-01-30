# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=i386-unknown-linux-gnu -position-independent \
# RUN:     -filetype=obj -o %t/elf_sm_pic_reloc_plt.o %s
# RUN: /home/ec2-user/llvm-project/build-32/bin/llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0xfff00000 -slab-page-size 4096 \
# RUN:     -abs external_func=0xffff0010 \
# RUN:     -check %s %t/elf_sm_pic_reloc_plt.o
#
# Test ELF small/PIC PLT relocations.

# Empty main entry point.
        .text
        .globl  main
        .p2align        4, 0x90
        .type   main,@function
main:
        ret
        .size   main, .-main

# Check R_386_PLT32 handling with a call to an external function via PLT. 
# This produces a Branch32 edge that is resolved like a regular PCRel32 
# (no PLT entry created).
# 
# NOTE - For ELF/i386 we always optimize away the PLT calls as the 
# displacement between the target address and the edge address always 
# fits in an int32_t. Regardless, we always create the PLT stub and GOT entry
# for position independent code, first, as there may be future use-cases
# where we would want to disable the optimization.
# 
# jitlink-check: decode_operand(test_call_extern_plt, 0) = external_func - next_pc(test_call_extern_plt)
# jitlink-check: *{4}(got_addr(elf_sm_pic_reloc_plt.o, external_func))= external_func
        .globl  test_call_extern_plt
        .p2align       4, 0x90
        .type   test_call_extern_plt,@function
test_call_extern_plt:
        call   external_func@plt

        .size   test_call_extern_plt, .-test_call_extern_plt