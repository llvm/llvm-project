# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=i386-unknown-linux-gnu -position-independent \
# RUN:  -filetype=obj -o %t/elf_sm_pic_reloc_got.o %s
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0xfff00000 -slab-page-size 4096 \
# RUN:     -check %s %t/elf_sm_pic_reloc_got.o
#
# Test ELF small/PIC GOT relocations.

        .text
        .globl  main
        .p2align        4, 0x90
        .type   main,@function
main:
        ret
        .size   main, .-main


# Test GOT32 handling.
# 
# We want to check both the offset to the GOT entry and its contents. 
# jitlink-check: decode_operand(test_got, 4) = got_addr(elf_sm_pic_reloc_got.o, named_data1) - _GLOBAL_OFFSET_TABLE_ + 42
# jitlink-check: *{4}(got_addr(elf_sm_pic_reloc_got.o, named_data1)) = named_data1
# 
# jitlink-check: decode_operand(test_got+6, 4) = got_addr(elf_sm_pic_reloc_got.o, named_data2) - _GLOBAL_OFFSET_TABLE_ + 5
# jitlink-check: *{4}(got_addr(elf_sm_pic_reloc_got.o, named_data2)) = named_data2

        .globl test_got
        .p2align      4, 0x90
        .type   test_got,@function
test_got:
        leal    named_data1@GOT+42, %eax
        leal    named_data2@GOT+5, %eax
        .size   test_got, .-test_got



# Test GOTOFF64 handling.
# jitlink-check: decode_operand(test_gotoff, 1) = named_func - _GLOBAL_OFFSET_TABLE_ + 99
        .globl test_gotoff
        .p2align     4, 0x90
        .type  test_gotoff,@function
test_gotoff:
        mov $named_func@GOTOFF+99, %eax
        .size   test_gotoff, .-test_gotoff


        .globl  named_func
        .p2align       4, 0x90
        .type   named_func,@function
named_func:
        xor    %eax, %eax
        .size   named_func, .-named_func


        .data

        .type   named_data1,@object
        .p2align        3
named_data1:
        .quad   42
        .size   named_data1, 8
        
        .type   named_data2,@object
        .p2align        3
named_data2:
        .quad   42
        .size   named_data2, 8
