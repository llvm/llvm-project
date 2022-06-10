# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=aarch64-unknown-linux-gnu -relax-relocations=false \
# RUN:   -position-independent -filetype=obj -o %t/elf_reloc.o %s
# RUN: llvm-jitlink -noexec \
# RUN:              -abs external_data=0xdeadbeef \
# RUN:              -check %s %t/elf_reloc.o

        .text

        .globl  main
        .p2align  2
        .type main,@function
main:
        ret

        .size   main, .-main

# Check R_AARCH64_CALL26 / R_AARCH64_JUMP26 relocation of a local function call
#
# jitlink-check: decode_operand(local_func_call26, 0)[25:0] = (local_func - local_func_call26)[27:2]
# jitlink-check: decode_operand(local_func_jump26, 0)[25:0] = (local_func - local_func_jump26)[27:2]
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

        .globl  local_func_jump26
        .p2align  2
local_func_jump26:
        b	local_func
        .size   local_func_jump26, .-local_func_jump26

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

# Check R_AARCH64_LDST*_ABS_LO12_NC relocation of a local symbol
#
# The immediate value should be the symbol address right shifted according to its instruction bitwidth.
#
# jitlink-check: decode_operand(test_ldrb, 2) = named_data[11:0]
# jitlink-check: decode_operand(test_ldrsb, 2) = (named_data + 0)[11:0]
# jitlink-check: decode_operand(test_ldrh, 2) = (named_data + 0)[11:1]
# jitlink-check: decode_operand(test_ldrsh, 2) = (named_data + 0)[11:1]
# jitlink-check: decode_operand(test_ldr_32bit, 2) = (named_data + 0)[11:2]
# jitlink-check: decode_operand(test_ldr_64bit, 2) = (named_data + 0)[11:3]
# jitlink-check: decode_operand(test_strb, 2) = named_data[11:0]
# jitlink-check: decode_operand(test_strh, 2) = (named_data + 0)[11:1]
# jitlink-check: decode_operand(test_str_32bit, 2) = (named_data + 0)[11:2]
# jitlink-check: decode_operand(test_str_64bit, 2) = (named_data + 0)[11:3]

        .globl  test_ldrb
test_ldrb:
        ldrb	w0, [x1, :lo12:named_data]
        .size test_ldrb, .-test_ldrb

        .globl  test_ldrsb
test_ldrsb:
        ldrsb	w0, [x1, :lo12:named_data]
        .size test_ldrsb, .-test_ldrsb

        .globl  test_ldrh
test_ldrh:
        ldrh	w0, [x1, :lo12:named_data]
        .size test_ldrh, .-test_ldrh

        .globl  test_ldrsh
test_ldrsh:
        ldrsh	w0, [x1, :lo12:named_data]
        .size test_ldrsh, .-test_ldrsh

        .globl  test_ldr_32bit
test_ldr_32bit:
        ldr	w0, [x1, :lo12:named_data]
        .size test_ldr_32bit, .-test_ldr_32bit

        .globl  test_ldr_64bit
test_ldr_64bit:
        ldr	x0, [x1, :lo12:named_data]
        .size test_ldr_64bit, .-test_ldr_64bit

        .globl  test_strb
test_strb:
        strb	w0, [x1, :lo12:named_data]
        .size test_strb, .-test_strb

        .globl  test_strh
test_strh:
        strh	w0, [x1, :lo12:named_data]
        .size test_strh, .-test_strh

        .globl  test_str_32bit
test_str_32bit:
        str	w0, [x1, :lo12:named_data]
        .size test_str_32bit, .-test_str_32bit

        .globl  test_str_64bit
test_str_64bit:
        str	x0, [x1, :lo12:named_data]
        .size test_str_64bit, .-test_str_64bit

# Check R_AARCH64_ABS64 relocation of a function pointer to local symbol
#
# jitlink-check: *{8}local_func_addr_quad = named_func
        .globl  local_func_addr_quad
        .p2align  3
local_func_addr_quad:
        .xword	named_func
        .size	local_func_addr_quad, 8

# Check R_AARCH64_ADR_GOT_PAGE / R_AARCH64_LD64_GOT_LO12_NC handling with a
# reference to an external symbol. Validate both the reference to the GOT entry,
# and also the content of the GOT entry.
#
# For the ADRP :got: instruction we have the 21-bit delta to the 4k page
# containing the GOT entry for external_data.
#
# For the LDR :got_lo12: instruction we have the 12-bit offset of the entry
# within the page.
#
# jitlink-check: *{8}(got_addr(elf_reloc.o, external_data)) = external_data
# jitlink-check: decode_operand(test_adr_gotpage_external, 1) = \
# jitlink-check:     (got_addr(elf_reloc.o, external_data)[32:12] - \
# jitlink-check:        test_adr_gotpage_external[32:12])
# jitlink-check: decode_operand(test_ld64_gotlo12_external, 2) = \
# jitlink-check:     got_addr(elf_reloc.o, external_data)[11:3]
        .globl  test_adr_gotpage_external
        .p2align  2
test_adr_gotpage_external:
        adrp  x0, :got:external_data
        .size test_adr_gotpage_external, .-test_adr_gotpage_external

        .globl  test_ld64_gotlo12_external
        .p2align  2
test_ld64_gotlo12_external:
        ldr   x0, [x0, :got_lo12:external_data]
        .size test_ld64_gotlo12_external, .-test_ld64_gotlo12_external

        .globl  named_data
        .p2align  4
        .type   named_data,@object
named_data:
        .quad   0x2222222222222222
        .quad   0x3333333333333333
        .size   named_data, .-named_data

        .globl  named_func
        .p2align  2
        .type	named_func,@function
named_func:
        ret
        .size   named_func, .-named_func
