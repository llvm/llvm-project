# RUN: llvm-mc -triple powerpc64le-unknown-linux-gnu %s -filetype=obj -o - | \
# RUN:    llvm-objdump -D  -r - | FileCheck %s

        .text
        .abiversion 2
        .global test
        .p2align 4
        .type test,@function
test:
.Lgep:
        addis 2, 12, .TOC.-.Lgep@ha
        addi  2,  2, .TOC.-.Lgep@l
.Llep:
        .localentry  test, .Llep-.Lgep
        addis 3, 2, vecA@toc@ha
        lxv   3,    vecA@toc@l(3)
        addis 3, 2, vecB@toc@ha
        stxv  3,    vecB@toc@l(3)
        blr

        .comm  vecA, 16, 16
        .comm  vecB, 16, 16

# CHECK: Disassembly of section .text:
# CHECK-LABEL: test:
# CHECK-NEXT:    addis 2, 12, 0
# CHECK-NEXT:    R_PPC64_REL16_HA     .TOC.
# CHECK-NEXT:    addi 2, 2, 0
# CHECK-NEXT:    R_PPC64_REL16_LO     .TOC.
# CHECK-NEXT:    addis 3, 2, 0
# CHECK-NEXT:    R_PPC64_TOC16_HA     vecA
# CHECK-NEXT:    lxv 3, 0(3)
# CHECK-NEXT:    R_PPC64_TOC16_LO_DS  vecA
# CHECK-NEXT:    addis 3, 2, 0
# CHECK-NEXT:    R_PPC64_TOC16_HA     vecB
# CHECK-NEXT:    stxv 3, 0(3)
# CHECK-NEXT:    R_PPC64_TOC16_LO_DS  vecB
# CHECK-NEXT:    blr

