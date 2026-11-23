# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc --triple=loongarch32 --filetype=obj -o %t/elf_reloc.o %s
# RUN: llvm-jitlink --noexec \
# RUN:              --abs external_data=0xdeadbeef \
# RUN:              --abs external_func=0xcafef00d \
# RUN:              --check %s %t/elf_reloc.o
    .text

    .globl main
    .p2align 2
    .type main,@function
main:
    ret

    .size main, .-main

## Check R_LARCH_B26 relocation of a local function call.

# jitlink-check: decode_operand(local_func_call26, 0)[27:0] = \
# jitlink-check:   (local_func - local_func_call26)[27:0]
# jitlink-check: decode_operand(local_func_jump26, 0)[27:0] = \
# jitlink-check:   (local_func - local_func_jump26)[27:0]
    .globl local_func
    .p2align 2
    .type local_func,@function
local_func:
    ret
    .size local_func, .-local_func

    .globl local_func_call26
    .p2align 2
local_func_call26:
    bl local_func
    .size local_func_call26, .-local_func_call26

    .globl local_func_jump26
    .p2align 2
local_func_jump26:
    b local_func
    .size local_func_jump26, .-local_func_jump26

## Check R_LARCH_PCADD_HI20 / R_LARCH_PCADD_LO12 relocation of a local symbol.

# jitlink-check: decode_operand(test_pcadd_hi20, 1)[19:0] = \
# jitlink-check:   (named_data - test_pcadd_hi20)[31:12] + \
# jitlink-check:      (named_data - test_pcadd_hi20)[11:11]
# jitlink-check: decode_operand(test_pcadd_lo12, 2)[11:0] = \
# jitlink-check:   (named_data - test_pcadd_hi20)[11:0]
    .globl test_pcadd_hi20
    .p2align 2
test_pcadd_hi20:
    pcaddu12i $a0, %pcadd_hi20(named_data)
    .size test_pcadd_hi20, .-test_pcadd_hi20

    .globl test_pcadd_lo12
    .p2align 2
test_pcadd_lo12:
    addi.w $a0, $a0, %pcadd_lo12(test_pcadd_hi20)
    .size test_pcadd_lo12, .-test_pcadd_lo12

## Check R_LARCH_PCALA_HI20 / R_LARCH_PCALA_LO12 relocation of a local symbol.

# jitlink-check: decode_operand(test_pc_hi20, 1)[19:0] = \
# jitlink-check:   (named_data - test_pc_hi20)[31:12] + \
# jitlink-check:      named_data[11:11]
# jitlink-check: decode_operand(test_pc_lo12, 2)[11:0] = \
# jitlink-check:   (named_data)[11:0]
    .globl test_pc_hi20
    .p2align 2
test_pc_hi20:
    pcalau12i $a0, %pc_hi20(named_data)
    .size test_pc_hi20, .-test_pc_hi20

    .globl test_pc_lo12
    .p2align 2
test_pc_lo12:
    addi.w $a0, $a0, %pc_lo12(named_data)
    .size test_pc_lo12, .-test_pc_lo12

## Check that calls/jumps to external functions trigger the generation of stubs
## and GOT entries.

# jitlink-check: *{4}(got_addr(elf_reloc.o, external_func)) = external_func
# jitlink-check: decode_operand(test_external_call, 0) = \
# jitlink-check:   (stub_addr(elf_reloc.o, external_func) - \
# jitlink-check:      test_external_call)[27:0]
# jitlink-check: decode_operand(test_external_jump, 0) = \
# jitlink-check:   (stub_addr(elf_reloc.o, external_func) - \
# jitlink-check:      test_external_jump)[27:0]
# jitlink-check: decode_operand(test_external_call30, 1)[19:0] = \
# jitlink-check:   (stub_addr(elf_reloc.o, external_func) - \
# jitlink-check:      test_external_call30)[31:12]
# jitlink-check: decode_operand(test_external_call30 + 4, 2)[17:0] = \
# jitlink-check:   (stub_addr(elf_reloc.o, external_func) - \
# jitlink-check:      test_external_call30)[11:0]
    .globl test_external_call
    .p2align  2
test_external_call:
    bl external_func
    .size test_external_call, .-test_external_call

    .globl test_external_jump
    .p2align 2
test_external_jump:
    b external_func
    .size test_external_jump, .-test_external_jump

    .globl test_external_call30
    .p2align  2
test_external_call30:
    pcaddu12i $ra, %call30(external_func)
    jirl $ra, $ra, 0
    .size test_external_call30, .-test_external_call30

## Check R_LARCH_GOT_PCADD_HI20 / R_LARCH_GOT_PCADD_LO12 handling with a
## reference to an external symbol. Validate both the reference to the GOT
## entry, and also the content of the GOT entry.

# jitlink-check: *{4}(got_addr(elf_reloc.o, external_data)) = external_data
# jitlink-check: decode_operand(test_got_pcadd_hi20_external, 1)[19:0] = \
# jitlink-check:   (got_addr(elf_reloc.o, external_data) - \
# jitlink-check:      test_got_pcadd_hi20_external)[31:12] + \
# jitlink-check:      (got_addr(elf_reloc.o, external_data) - \
# jitlink-check:         test_got_pcadd_hi20_external)[11:11]
# jitlink-check: decode_operand(test_got_pcadd_lo12_external, 2)[11:0] = \
# jitlink-check:   (got_addr(elf_reloc.o, external_data) - \
# jitlink-check:      test_got_pcadd_hi20_external)[11:0]
    .globl test_got_pcadd_hi20_external
    .p2align 2
test_got_pcadd_hi20_external:
    pcaddu12i $a0, %got_pcadd_hi20(external_data)
    .size test_got_pcadd_hi20_external, .-test_got_pcadd_hi20_external

    .globl test_got_pcadd_lo12_external
    .p2align 2
test_got_pcadd_lo12_external:
    ld.w $a0, $a0, %got_pcadd_lo12(test_got_pcadd_hi20_external)
    .size test_got_pcadd_lo12_external, .-test_got_pcadd_lo12_external

## Check R_LARCH_GOT_PC_HI20 / R_LARCH_GOT_PC_LO12 handling with a reference to
## an external symbol. Validate both the reference to the GOT entry, and also
## the content of the GOT entry.

# jitlink-check: *{4}(got_addr(elf_reloc.o, external_data)) = external_data
# jitlink-check: decode_operand(test_got_pc_hi20_external, 1)[19:0] = \
# jitlink-check:   (got_addr(elf_reloc.o, external_data)[31:12] - \
# jitlink-check:      test_got_pc_hi20_external[31:12] + \
# jitlink-check:      got_addr(elf_reloc.o, external_data)[11:11])[19:0]
# jitlink-check: decode_operand(test_got_pc_lo12_external, 2)[11:0] = \
# jitlink-check:   got_addr(elf_reloc.o, external_data)[11:0]
    .globl test_got_pc_hi20_external
    .p2align 2
test_got_pc_hi20_external:
    pcalau12i $a0, %got_pc_hi20(external_data)
    .size test_got_pc_hi20_external, .-test_got_pc_hi20_external

    .globl test_got_pc_lo12_external
    .p2align 2
test_got_pc_lo12_external:
    ld.w $a0, $a0, %got_pc_lo12(external_data)
    .size test_got_pc_lo12_external, .-test_got_pc_lo12_external

## Check R_LARCH_CALL30 relocation of a local function call.

# jitlink-check: decode_operand(local_func_call30, 1)[19:0] = \
# jitlink-check:   (local_func - local_func_call30)[31:12]
# jitlink-check: decode_operand(local_func_call30 + 4, 2)[17:0] = \
# jitlink-check:   (local_func - local_func_call30)[11:0]
    .globl local_func_call30
    .p2align 2
local_func_call30:
    pcaddu12i $ra, %call30(local_func)
    jirl $ra, $ra, 0
    .size local_func_call30, .-local_func_call30

## Check R_LARCH_B16 relocation for compare and branch instructions.

# jitlink-check: decode_operand(test_br16, 2)[17:0] = \
# jitlink-check:   (test_br16_target - test_br16)[17:0]
    .globl test_br16, test_br16_target
    .p2align 2
test_br16:
    beq $t1, $t2, %b16(test_br16_target)
    .skip (1 << 16)
test_br16_target:
    .size test_br16, .-test_br16

## Check R_LARCH_B21 relocation for compare and branch instructions.

# jitlink-check: decode_operand(test_br21, 1)[22:0] = \
# jitlink-check:   (test_br21_target - test_br21)[22:0]
    .globl test_br21, test_br21_target
    .p2align 2
test_br21:
    beqz $t1, %b21(test_br21_target)
    .skip (1 << 21)
test_br21_target:
    .size test_br21, .-test_br21


    .globl named_data
    .p2align 4
    .type named_data,@object
named_data:
    .quad 0x2222222222222222
    .quad 0x3333333333333333
    .size named_data, .-named_data
