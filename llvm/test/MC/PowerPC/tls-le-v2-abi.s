// RUN: llvm-mc -triple=powerpc64le-pc-linux -filetype=obj %s -o - | \
// RUN: llvm-readobj -r | FileCheck %s

// RUN: llvm-mc -triple=powerpc64-pc-linux -filetype=obj %s -o - | \
// RUN: llvm-readobj -r | FileCheck %s

// Verify we can handle all the tprel symbol modifiers for local exec tls.
// Tests 16 bit offsets on both DS-form and D-form instructions, 32 bit
// adjusted and non-adjusted offsets and 64 bit adjusted and non-adjusted
// offsets.
        .text
        .abiversion 2

        .globl	short_offset_ds
        .p2align	4
        .type	short_offset_ds,@function
short_offset_ds:
        lwa 3, i@tprel(13)
        blr

        .globl short_offset
        .p2align        4
        .type   short_offset,@function
short_offset:
        addi 3, 13, i@tprel
        blr

        .globl	medium_offset
        .p2align	4
        .type	medium_offset,@function
medium_offset:
        addis 3, 13, i@tprel@ha
        lwa 3, i@tprel@l(3)
        blr

        .globl  medium_not_adjusted
        .p2align        4
        .type   medium_not_adjusted,@function
medium_not_adjusted:
        lis 3, i@tprel@h
        ori 3, 3, i@tprel@l
        lwax 3, 3, 13
        blr

        .globl	large_offset
        .p2align	4
        .type	large_offset,@function
large_offset:
        lis 3, i@tprel@highesta
        ori 3, 3, i@tprel@highera
        sldi 3, 3, 32
        oris 3, 3, i@tprel@higha
        addi  3, 3, i@tprel@l
        lwax 3, 3, 13
        blr

        .globl	not_adjusted
        .p2align	4
        .type	not_adjusted,@function
not_adjusted:
        lis 3, i@tprel@highest
        ori 3, 3, i@tprel@higher
        sldi 3, 3, 32
        oris 3, 3, i@tprel@high
        ori  3, 3, i@tprel@l
        lwax 3, 3, 13
        blr

        .type	i,@object
        .section	.tdata,"awT",@progbits
        .p2align	2
i:
        .long	55
        .size	i, 4

        .type j,@object
        .data
        .p2align        3
j:
        .quad i@tprel
        .size j, 8


# CHECK: Relocations [
# CHECK:   Section {{.*}} .rela.text {
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_TPREL16_DS i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_TPREL16 i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_TPREL16_HA i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_TPREL16_LO_DS i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_TPREL16_HI i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_TPREL16_LO i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_TPREL16_HIGHESTA i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_TPREL16_HIGHERA i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_TPREL16_HIGHA i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_TPREL16_LO i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_TPREL16_HIGHEST i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_TPREL16_HIGHER i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_TPREL16_HIGH i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_TPREL16_LO i
# CHECK:   }
# CHECK:  Section (6) .rela.data {
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_TPREL64 i
# CHECK:   }
# CHECK: ]
