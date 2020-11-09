# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-unknown-linux -position-independent -filetype=obj -o %t/elf_common.o %s
# RUN: llvm-jitlink -entry=load_common -noexec -check %s %t/elf_common.o

        .text
        .file   "elf_common.c"
        .globl  load_common
        .p2align        4, 0x90
        .type   load_common,@function

load_common:
# Check that common variable GOT entry is synthesized correctly. In order to
# prevent the optimizer from relaxing the edge, we use a movl instruction.
# jitlink-check: decode_operand(load_common, 4) = \
# jitlink-check:   got_addr(elf_common.o, common_data) - next_pc(load_common)
# jitlink-check: *{8}(got_addr(elf_common.o, common_data)) = common_data
        movl    common_data@GOTPCREL(%rip), %eax
        ret

        .size   load_common, .-load_common

# Check that common is zero-filled.
# jitlink-check: *{4}(common_data) = 0
        .type   common_data,@object
        .comm   common_data,4,4
