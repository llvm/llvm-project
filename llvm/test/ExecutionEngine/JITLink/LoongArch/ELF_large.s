# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc --triple=loongarch64 --filetype=obj -o %t/elf_large_reloc.o %s
# RUN: llvm-jitlink --noexec \
# RUN:              --abs external_data=0x1234 \
# RUN:              --check %s %t/elf_large_reloc.o

    .text
    .globl main
    .p2align 2
    .type main,@function
main:
    ret
    .size main,.-main

    .section .sec.large.pc,"ax",@progbits
    .globl test_large_pc
    .p2align 2
test_large_pc:
# jitlink-check: *{4}(test_large_pc) = 0x1bffffed
# jitlink-check: *{4}(test_large_pc + 4) = 0x2c0000c
# jitlink-check: *{4}(test_large_pc + 8) = 0x1600000c
# jitlink-check: *{4}(test_large_pc + 12) = 0x300018c
    pcalau12i $t1, %pc_hi20(named_data)        # R_LARCH_PCALA_HI20
    addi.d $t0, $zero, %pc_lo12(named_data)    # R_LARCH_PCALA_LO12
    lu32i.d $t0, %pc64_lo20(named_data)        # R_LARCH_PCALA64_LO20
    lu52i.d $t0, $t0, %pc64_hi12(named_data)   # R_LARCH_PCALA64_HI12
    .size test_large_pc, .-test_large_pc

    .section .sec.large.got,"ax",@progbits
    .globl test_large_got
    .p2align 2
test_large_got:
# jitlink-check: *{4}(test_large_got) = 0x1a00000d
# jitlink-check: *{4}(test_large_got + 4) = 0x2c0a00c
# jitlink-check: *{4}(test_large_got + 8) = 0x1600000c
# jitlink-check: *{4}(test_large_got + 12) = 0x300018c
    pcalau12i $t1, %got_pc_hi20(external_data)      # R_LARCH_GOT_PC_HI20
    addi.d $t0, $zero, %got_pc_lo12(external_data)  # R_LARCH_GOT_PC_LO12
    lu32i.d $t0, %got64_pc_lo20(external_data)      # R_LARCH_GOT64_PC_LO20
    lu52i.d $t0, $t0, %got64_pc_hi12(external_data) # R_LARCH_GOT64_PC_HI12
    .size test_large_got, .-test_large_got

    .data
    .globl named_data
    .p2align 4
    .type named_data,@object
named_data:
    .quad 0x1111111111111111
    .quad 0x2222222222222222
    .size named_data, .-named_data
