# REQUIRES: system-linux
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=systemz-unknown-linux -position-independent \
# RUN:     -filetype=obj -o  %t/elf_pic_reloc.o %s 
#
# RUN: llvm-jitlink -noexec \ 
# RUN:     -slab-allocate 100Kb -slab-address 0xfff00000 -slab-page-size 4096 \
# RUN:     -abs external_data=0x1 \
# RUN:     -abs extern_out_of_range32=0x7fff00000000 \
# RUN:     -abs extern_in_range32=0xffe00000 \
# RUN:     -check %s %t/elf_pic_reloc.o

        .text
        .section        .text.main
        .globl  main
        .p2align        4
        .type   main,@function
main:
         br   %r14
         .size main, .-main

        .globl  named_func
        .p2align       4
        .type   named_func,@function
named_func:
	br    %r14
        .size   named_func, .-named_func

# Check R_390_PLT32DBL handling with a call to a local function in the text
# section. This produces a Branch32 edge that is resolved like a regular
# BranchPCRelPLT32dbl(no PLT entry created).
#
# jitlink-check: decode_operand(test_call_local, 1) = \
# jitlink-check:   named_func - test_call_local
        .globl  test_call_local
        .p2align       4
        .type   test_call_local,@function
test_call_local:
        brasl  %r14, named_func@PLT 

        .size   test_call_local, .-test_call_local

# Check R_390_PLT32dbl(BranchPCRelPLT32dbl)  handling with a call to an 
# external via PLT. This produces a Branch32ToStub edge, because externals are 
# not defined locally. As the target is out-of-range from the callsite, 
# the edge keeps using its PLT entry.
#
# jitlink-check: decode_operand(test_call_extern_plt, 1) = \
# jitlink-check:     stub_addr(elf_pic_reloc.o, extern_out_of_range32) - \
# jitlink-check:        test_call_extern_plt
# jitlink-check: *{8}(got_addr(elf_pic_reloc.o, extern_out_of_range32)) = \
# jitlink-check:     extern_out_of_range32
        .globl  test_call_extern_plt
        .p2align       4
        .type   test_call_extern_plt,@function
test_call_extern_plt:
        brasl   %r14, extern_out_of_range32@plt

        .size   test_call_extern_plt, .-test_call_extern_plt

# Check R_390_PLT32(BranchPCRelPLT32dbl) handling with a call to an external. 
# This produces a Branch32ToStub edge, because externals are not defined 
# locally. During resolution, the target turns out to be in-range from the 
# callsite.
### TODO: edge can be relaxed in post-allocation optimization, it will then
### require:
### jitlink-check: decode_operand(test_call_extern, 1) = \
### jitlink-check:     extern_in_range32 - test_call_extern
#
# Same as test_call_extern_plt(no-optimization)
# jitlink-check: decode_operand(test_call_extern, 1) = \
# jitlink-check:     stub_addr(elf_pic_reloc.o, extern_in_range32) - \
# jitlink-check:        test_call_extern
# jitlink-check: *{8}(got_addr(elf_pic_reloc.o, extern_in_range32)) = \
# jitlink-check:     extern_in_range32
        .globl  test_call_extern
        .p2align       4
        .type   test_call_extern,@function
test_call_extern:
        brasl   %r14, extern_in_range32@plt
        .size   test_call_extern, .-test_call_extern

