# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=aarch64-unknown-linux-gnu -relax-relocations=false -position-independent -filetype=obj -o %t/elf_reloc.o %s
# RUN: llvm-jitlink -noexec -check %s %t/elf_reloc.o

        .text

        .globl  main
        .p2align  2
        .type main,@function
main:
        ret

        .size   main, .-main

# Check R_AARCH64_CALL26 relocation of a local function call
#
# jitlink-check: decode_operand(local_func_call26, 0)[25:0] = (local_func - local_func_call26)[27:2]
        .globl  local_func
        .p2align  2
        .type	local_func,@function
local_func:
        ret
        .size   local_func, .-local_func

        .globl  local_func_call26
        .p2align  2
local_func_call26:
        bl	local_func
        .size   local_func_call26, .-local_func_call26


# Check R_AARCH64_ADR_PREL_PG_HI21 / R_AARCH64_ADD_ABS_LO12_NC relocation of a local symbol
#
# For the ADR_PREL_PG_HI21/ADRP instruction we have the 21-bit delta to the 4k page
# containing the global.
#
# jitlink-check: decode_operand(test_adr_prel, 1) = (named_data - test_adr_prel)[32:12]
# jitlink-check: decode_operand(test_add_abs_lo12, 2) = (named_data + 0)[11:0]
        .globl  test_adr_prel
        .p2align  2
test_adr_prel:
        adrp	x0, named_data
        .size test_adr_prel, .-test_adr_prel

        .globl  test_add_abs_lo12
        .p2align  2
test_add_abs_lo12:
        add	x0, x0, :lo12:named_data
        .size test_add_abs_lo12, .-test_add_abs_lo12

        .globl  named_data
        .p2align  4
        .type   named_data,@object
named_data:
        .quad   0x2222222222222222
        .quad   0x3333333333333333
        .size   named_data, .-named_data
