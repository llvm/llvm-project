# RUN: llvm-mc --triple=aarch64-elf --filetype=obj < %s | \
# RUN:   llvm-objdump -d -r --symbolize-operands --no-show-raw-insn --no-leading-addr - | \
# RUN:   FileCheck %s --match-full-lines

# CHECK:      <fn1>:
# CHECK-NEXT:   b <L0>
# CHECK-NEXT:   tbz x0, #0x2c, <L2>
# CHECK-NEXT: <L0>:
# CHECK-NEXT:   b.eq <L1>
# CHECK-NEXT: <L1>:
# CHECK-NEXT:   cbz x1, <L0>
# CHECK-NEXT: <L2>:
# CHECK-NEXT:   nop
# CHECK-NEXT: <L3>:
# CHECK-NEXT:   bl <L3>
# CHECK-NEXT:     R_AARCH64_CALL26 fn2
# CHECK-NEXT:   bl <fn2>
# CHECK-NEXT:   adr x0, <L2>
# CHECK-NEXT: <L4>:
# CHECK-NEXT:   adr x1, <L4>
# CHECK-NEXT:     R_AARCH64_ADR_PREL_LO21 fn2
# CHECK-NEXT:   adr x2, <fn2>
# CHECK-NEXT:   ldr w0, <L2>
# CHECK-NEXT: <L5>:
# CHECK-NEXT:   ldr w0, <L5>
# CHECK-NEXT:     R_AARCH64_LD_PREL_LO19 fn2
# CHECK-NEXT:   ret
# CHECK-NEXT:   nop
# CHECK-NEXT:   nop
# CHECK-NEXT:   nop
# CHECK-EMPTY:
# CHECK-NEXT: <fn2>:
# CHECK-NEXT:   bl <L0>
# CHECK-NEXT:   adrp x3, 0x0 <fn1>
# CHECK-NEXT:     R_AARCH64_ADR_PREL_PG_HI21 fn2
# CHECK-NEXT:   add x3, x3, #0x0
# CHECK-NEXT:     R_AARCH64_ADD_ABS_LO12_NC fn2
# CHECK-NEXT:   adrp x3, 0x0 <fn1>
# CHECK-NEXT:     R_AARCH64_ADR_PREL_PG_HI21 fn2
# CHECK-NEXT:   ldr x0, [x3]
# CHECK-NEXT:     R_AARCH64_LDST64_ABS_LO12_NC fn2
# CHECK-NEXT:   ret
# CHECK-NEXT:   nop
# CHECK-NEXT:   nop
# CHECK-NEXT: <L0>:
# CHECK-NEXT:   ret

    .p2align 4
    .global fn1
fn1:
    b 0f
    tbz x0, 44, 2f
0:  b.eq 1f
1:  cbz x1, 0b
2:  nop
    bl fn2
    bl .Lfn2
    adr x0, 2b
    adr x1, fn2
    adr x2, .Lfn2
    ldr w0, 2b
    ldr w0, fn2
    ret

    .p2align 4
    .global fn2
fn2:
.Lfn2: ## Local label for non-interposable call.
    bl .Lfn3
    ## In future, we might identify the pairs and symbolize the operands properly.
    adrp x3, fn2
    add x3, x3, :lo12:fn2
    adrp x3, fn2
    ldr x0, [x3, :lo12:fn2]
    ret

    .p2align 4
.Lfn3: ## Private function
    ret
