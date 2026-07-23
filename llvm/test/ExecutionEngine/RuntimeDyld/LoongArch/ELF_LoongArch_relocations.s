# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc --triple=loongarch64 --filetype=obj -o %t/reloc.o %s
# RUN: llvm-rtdyld --triple=loongarch64 --verify --check=%s %t/reloc.o \
# RUN:     --map-section reloc.o,.got=0x21f00 \
# RUN:     --map-section reloc.o,.sec.large.pc=0x0000000012345000 \
# RUN:     --map-section reloc.o,.sec.large.got=0x44433333abcde000 \
# RUN:     --map-section reloc.o,.sec.dummy=0x4443333334567111 \
# RUN:     --dummy-extern abs=0x0123456789abcdef \
# RUN:     --dummy-extern external_data=0x1234

    .text
    .globl main
    .p2align 2
    .type   main,@function
main:
## Check R_LARCH_ABS_HI20
# rtdyld-check: *{4}(main) = 0x1513578c
    lu12i.w $t0, %abs_hi20(abs)
## Check R_LARCH_ABS_LO12
# rtdyld-check: *{4}(main + 4) = 0x03b7bd8c
    ori $t0, $t0, %abs_lo12(abs)
## Check R_LARCH_ABS64_LO20
# rtdyld-check: *{4}(main + 8) = 0x1668acec
    lu32i.d $t0, %abs64_lo20(abs)
## Check R_LARCH_ABS64_HI12
# rtdyld-check: *{4}(main + 12) = 0x0300498c
    lu52i.d $t0, $t0, %abs64_hi12(abs)
		ret
	  .size main, .-main

    .globl local_func
    .p2align 2
    .type local_func,@function
local_func:
    ret
    .size local_func, .-local_func

    .globl local_func_call26
    .p2align 2
local_func_call26:
## Check R_LARCH_B26
# rtdyld-check: decode_operand(local_func_call26, 0)[27:0] = \
# rtdyld-check:   (local_func - local_func_call26)[27:0]
    bl local_func
    .size local_func_call26, .-local_func_call26

    .globl local_func_call36
    .p2align 2
local_func_call36:
## Check R_LARCH_CALL36
# rtdyld-check: decode_operand(local_func_call36, 1)[19:0] = \
# rtdyld-check:   ((local_func - local_func_call36) + \
# rtdyld-check:    (((local_func - local_func_call36)[17:17]) << 17))[37:18]
# rtdyld-check: decode_operand(local_func_call36 + 4, 2)[17:0] = \
# rtdyld-check:   (local_func - local_func_call36)[17:0]
    pcaddu18i $ra, %call36(local_func)
    jirl $ra, $ra, 0
    .size local_func_call36, .-local_func_call36

    .globl test_pc_hi20
    .p2align 2
test_pc_hi20:
## Check R_LARCH_PCALA_HI20
# rtdyld-check: decode_operand(test_pc_hi20, 1)[19:0] = \
# rtdyld-check:   (named_data - test_pc_hi20)[31:12] + \
# rtdyld-check:      named_data[11:11]
    pcalau12i $a0, %pc_hi20(named_data)
    .size test_pc_hi20, .-test_pc_hi20

    .globl test_pc_lo12
    .p2align 2
test_pc_lo12:
## Check R_LARCH_PCALA_LO12
# rtdyld-check: decode_operand(test_pc_lo12, 2)[11:0] = \
# rtdyld-check:   (named_data)[11:0]
    addi.d $a0, $a0, %pc_lo12(named_data)
    .size test_pc_lo12, .-test_pc_lo12

    .globl test_got_pc_hi20
    .p2align 2
test_got_pc_hi20:
## Check R_LARCH_GOT_PC_HI20
# rtdyld-check: decode_operand(test_got_pc_hi20, 1)[19:0] = \
# rtdyld-check:   (section_addr(reloc.o, .got)[31:12] - \
# rtdyld-check:    test_got_pc_hi20[31:12] + \
# rtdyld-check:    section_addr(reloc.o, .got)[11:11])
    pcalau12i $a0, %got_pc_hi20(external_data)
    .size test_got_pc_hi20, .-test_got_pc_hi20

    .globl test_got_pc_lo12
    .p2align 2
test_got_pc_lo12:
## Check R_LARCH_GOT_PC_LO12
# rtdyld-check: decode_operand(test_got_pc_lo12, 2)[11:0] = \
# rtdyld-check:   (section_addr(reloc.o, .got)[11:0])
    ld.d $a0, $a0, %got_pc_lo12(external_data)
    .size test_gotoffset12_external, .-test_gotoffset12_external

    .globl named_data
    .p2align 4
    .type named_data,@object
named_data:
    .quad 0x2222222222222222
    .quad 0x3333333333333333
    .size named_data, .-named_data

    .section .sec.large.pc,"ax"
    .globl test_large_pc
test_large_pc:
## Code after link should be:
## 1a44444d pcalau12i $t1, 139810
## 02c4440c addi.d    $t0, $zero, 273
## 1666666c lu32i.d   $t0, 209715
## 0311118c lu52i.d   $t0, $t0, 1092

# rtdyld-check: *{4}(test_large_pc) = 0x1a44444d
    pcalau12i $t1, %pc_hi20(.sec.dummy)
# rtdyld-check: *{4}(test_large_pc + 4) = 0x02c4440c
    addi.d $t0, $zero, %pc_lo12(.sec.dummy)
# rtdyld-check: *{4}(test_large_pc + 8) = 0x1666666c
    lu32i.d $t0, %pc64_lo20(.sec.dummy)
# rtdyld-check: *{4}(test_large_pc + 12) = 0x0311118c
    lu52i.d $t0, $t0, %pc64_hi12(.sec.dummy)

    .section .sec.large.got,"ax"
    .globl test_large_got
test_large_got:
## Code after link should be:
## 1aa8688d pcalau12i $t1, 344900
## 02fc000c addi.d    $t0, $zero, -256
## 1799996c lu32i.d   $t0, -209717
## 032eed8c lu52i.d   $t0, $t0, -1093

# rtdyld-check: *{4}(test_large_got) = 0x1aa8688d
    pcalau12i $t1, %got_pc_hi20(external_data)
# rtdyld-check: *{4}(test_large_got + 4) = 0x02fc000c
    addi.d $t0, $zero, %got_pc_lo12(external_data)
# rtdyld-check: *{4}(test_large_got + 8) = 0x1799996c
    lu32i.d $t0, %got64_pc_lo20(external_data)
# rtdyld-check: *{4}(test_large_got + 12) = 0x032eed8c
    lu52i.d $t0, $t0, %got64_pc_hi12(external_data)

    .section .sec.dummy,"a"
    .word 0
