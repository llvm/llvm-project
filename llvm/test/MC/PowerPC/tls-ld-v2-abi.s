// RUN: llvm-mc -triple=powerpc64le-pc-linux -filetype=obj %s -o - | \
// RUN: llvm-readobj -r - | FileCheck %s

// RUN: llvm-mc -triple=powerpc64-pc-linux -filetype=obj %s -o - | \
// RUN: llvm-readobj -r - | FileCheck %s

// Verify we can handle all the dtprel symbol modifiers for local-dynamic tls.
// Tests a 16 bit offset on both DS-form and D-form instructions, 32 bit
// adjusted and non-adjusted offsets, and 64 bit adjusted and non-adjusted
// offsets.
        .text
        .abiversion 2

        .globl	short_offset
        .p2align	4
        .type	short_offset,@function
short_offset:
.Lfunc_gep0:
        addis 2, 12, .TOC.-.Lfunc_gep0@ha
        addi 2, 2, .TOC.-.Lfunc_gep0@l
.Lfunc_lep0:
.localentry	short_offset, .Lfunc_lep0-.Lfunc_gep0
        mflr 0
        std 0, 16(1)
        stdu 1, -32(1)
        addis 3, 2, i@got@tlsld@ha
        addi 3, 3, i@got@tlsld@l
        bl __tls_get_addr(i@tlsld)
        nop
        lwa 4, i@dtprel(3)
        addi 5, 3, i@dtprel
        lwa 3, 0(5)
        add 3, 4, 3
        addi 1, 1, 32
        ld 0, 16(1)
        mtlr 0
        blr

        .globl	medium_offset
        .p2align	4
        .type	medium_offset,@function
medium_offset:
.Lfunc_gep1:
        addis 2, 12, .TOC.-.Lfunc_gep1@ha
        addi 2, 2, .TOC.-.Lfunc_gep1@l
.Lfunc_lep1:
        .localentry	medium_offset, .Lfunc_lep1-.Lfunc_gep1
        mflr 0
        std 0, 16(1)
        stdu 1, -32(1)
        addis 3, 2, i@got@tlsld@ha
        addi 3, 3, i@got@tlsld@l
        bl __tls_get_addr(i@tlsld)
        nop
        addis 3, 3, i@dtprel@ha
        lwa 3, i@dtprel@l(3)
        addi 1, 1, 32
        ld 0, 16(1)
        mtlr 0
        blr

        .globl  medium_not_adjusted
        .p2align        4
        .type   medium_not_adjusted,@function
medium_not_adjusted:
.Lfunc_gep2:
        addis 2, 12, .TOC.-.Lfunc_gep2@ha
        addi 2, 2, .TOC.-.Lfunc_gep2@l
.Lfunc_lep2:
        .localentry     medium_not_adjusted, .Lfunc_lep2-.Lfunc_gep2
        mflr 0
        std 0, 16(1)
        stdu 1, -32(1)
        addis 3, 2, i@got@tlsld@ha
        addi 3, 3, i@got@tlsld@l
        bl __tls_get_addr(i@tlsld)
        nop
        lis 4, i@dtprel@h
        ori 4, 4, i@dtprel@l
        add 3, 3, 4
        addi 1, 1, 32
        ld 0, 16(1)
        mtlr 0
        blr

        .globl	large_offset
        .p2align	4
        .type	large_offset,@function
large_offset:
.Lfunc_gep3:
        addis 2, 12, .TOC.-.Lfunc_gep3@ha
        addi 2, 2, .TOC.-.Lfunc_gep3@l
.Lfunc_lep3:
        .localentry	large_offset, .Lfunc_lep3-.Lfunc_gep3
        mflr 0
        std 0, 16(1)
        stdu 1, -32(1)
        addis 3, 2, i@got@tlsld@ha
        addi 3, 3, i@got@tlsld@l
        bl __tls_get_addr(i@tlsld)
        nop
        lis 4, i@dtprel@highesta
        ori 4, 4, i@dtprel@highera
        sldi 4, 4, 32
        addis 4, 4, i@dtprel@higha
        addi  4, 4, i@dtprel@l
        lwax 3, 4, 3
        addi 1, 1, 32
        ld 0, 16(1)
        mtlr 0
        blr

         .globl not_adjusted
         .p2align       4
         .type not_adjusted,@function
not_adjusted:
.Lfunc_gep4:
        addis 2, 12, .TOC.-.Lfunc_gep4@ha
        addi 2, 2, .TOC.-.Lfunc_gep4@l
.Lfunc_lep4:
        .localentry	not_adjusted, .Lfunc_lep4-.Lfunc_gep4
        mflr 0
        std 0, 16(1)
        stdu 1, -32(1)
        addis 3, 2, i@got@tlsld@ha
        addi 3, 3, i@got@tlsld@l
        bl __tls_get_addr(i@tlsld)
        nop
        lis 4, i@dtprel@highest
        ori 4, 4, i@dtprel@higher
        sldi 4, 4, 32
        oris 4, 4, i@dtprel@high
        ori  4, 4, i@dtprel@l
        lwax 3, 4, 3
        addi 1, 1, 32
        ld 0, 16(1)
        mtlr 0
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
        .quad i@dtprel
        .size j, 8

# CHECK: Relocations [
# CHECK:   Section {{.*}} .rela.text {
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_GOT_TLSLD16_HA i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_GOT_TLSLD16_LO i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_TLSLD i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_REL24 __tls_get_addr
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_DTPREL16_DS i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_DTPREL16 i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_DTPREL16_HA i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_DTPREL16_LO_DS i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_DTPREL16_HI i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_DTPREL16_HIGHESTA i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_DTPREL16_HIGHERA i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_DTPREL16_HIGHA i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_DTPREL16_LO i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_DTPREL16_HIGHEST i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_DTPREL16_HIGHER i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_DTPREL16_HIGH i
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_DTPREL16_LO i
# CHECK: }
# CHECK: Section {{.*}} .rela.data {
# CHECK: 0x{{[0-9A-F]+}}  R_PPC64_DTPREL64 i
# CHECK:   }
# CHECK: ]
