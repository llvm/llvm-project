# RUN: llvm-mc -triple=powerpc64le -filetype=obj %s | \
# RUN: llvm-objdump -dr --mcpu=pwr10 - | FileCheck %s
# RUN: llvm-mc -triple=powerpc64 -filetype=obj %s | \
# RUN: llvm-objdump -dr --mcpu=pwr10 - | FileCheck %s


##
# This section of tests contains the MCBinaryExpr as the first parameter of the
# .reloc relocation.
##
	.text
	.abiversion 2
	.globl	Minimal
	.p2align	4
	.type	Minimal,@function
Minimal:
.LMinimal$local:
	pld 3, vec@got@pcrel(0), 1
.Lpcrel1:
	.reloc .Lpcrel1-8,R_PPC64_PCREL_OPT,.-(.Lpcrel1-8)
	lwa 3, 4(3)
	blr
	.long	0
	.quad	0
# CHECK-LABEL:   Minimal
# CHECK:         pld 3, 0(0), 1
# CHECK-NEXT:    R_PPC64_GOT_PCREL34	vec
# CHECK-NEXT:    R_PPC64_PCREL_OPT	*ABS*+0x8
# CHECK-NEXT:    lwa 3, 4(3)
# CHECK-NEXT:    blr

	.globl	SingleInsnBetween
	.p2align	4
	.type	SingleInsnBetween,@function
SingleInsnBetween:
.LSingleInsnBetween$local:
	pld 3, vec@got@pcrel(0), 1
.Lpcrel2:
        addi 3, 3, 42
	.reloc .Lpcrel2-8,R_PPC64_PCREL_OPT,.-(.Lpcrel2-8)
	lwa 3, 4(3)
	blr
	.long	0
	.quad	0
# CHECK_LABEL:   SingleInsnBetween
# CHECK:         pld 3, 0(0), 1
# CHECK-NEXT:    R_PPC64_GOT_PCREL34	vec
# CHECK-NEXT:    R_PPC64_PCREL_OPT	*ABS*+0xc
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    lwa 3, 4(3)
# CHECK-NEXT:    blr


	.globl	MultiInsnBetween                    # -- Begin function
	.p2align	4
	.type	MultiInsnBetween,@function
MultiInsnBetween:
.LMultiInsnBetween$local:
	pld 3, vec@got@pcrel(0), 1
.Lpcrel3:
        addi 3, 3, 42
        addi 3, 3, 42
        addi 3, 3, 42
        addi 3, 3, 42
        addi 3, 3, 42
	.reloc .Lpcrel3-8,R_PPC64_PCREL_OPT,.-(.Lpcrel3-8)
	lwa 3, 4(3)
	blr
	.long	0
	.quad	0
# CHECK_LABEL:   MultiInsnBetween
# CHECK:         pld 3, 0(0), 1
# CHECK-NEXT:    R_PPC64_GOT_PCREL34	vec
# CHECK-NEXT:    R_PPC64_PCREL_OPT	*ABS*+0x1c
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    lwa 3, 4(3)
# CHECK-NEXT:    blr

	.globl	PrefixInsnBetween
	.p2align	6
	.type	PrefixInsnBetween,@function
        .space          48       # Add a space to force an alignment of a paddi.
PrefixInsnBetween:
.LPrefixInsnBetween$local:
	pld 3, vec@got@pcrel(0), 1
.Lpcrel4:
        addi 3, 3, 42
        paddi 3, 3, 42, 0
        addi 3, 3, 42
        paddi 3, 3, 42, 0
        addi 3, 3, 42
	.reloc .Lpcrel4-8,R_PPC64_PCREL_OPT,.-(.Lpcrel4-8)
	lwa 3, 4(3)
	blr
	.long	0
	.quad	0
# CHECK_LABEL:   PrefixInsnBetween
# CHECK:         pld 3, 0(0), 1
# CHECK-NEXT:    R_PPC64_GOT_PCREL34	vec
# CHECK-NEXT:    R_PPC64_PCREL_OPT	*ABS*+0x28
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    nop
# CHECK-NEXT:    paddi 3, 3, 42, 0
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    paddi 3, 3, 42, 0
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    lwa 3, 4(3)
# CHECK-NEXT:    blr


	.globl	SpaceBetween                    # -- Begin function
	.p2align	4
	.type	SpaceBetween,@function
SpaceBetween:
.LSpaceBetween$local:
	pld 3, vec@got@pcrel(0), 1
.Lpcrel5:
        addi 3, 3, 42
        paddi 3, 3, 42, 0
        addi 3, 3, 42
        .space 40, 0
        paddi 3, 3, 42, 0
        addi 3, 3, 42
	.reloc .Lpcrel5-8,R_PPC64_PCREL_OPT,.-(.Lpcrel5-8)
	lwa 3, 4(3)
	blr
	.long	0
	.quad	0
# CHECK_LABEL:   SpaceBetween
# CHECK:         pld 3, 0(0), 1
# CHECK-NEXT:    R_PPC64_GOT_PCREL34	vec
# CHECK-NEXT:    R_PPC64_PCREL_OPT	*ABS*+0x50
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    nop
# CHECK-NEXT:    paddi 3, 3, 42, 0
# CHECK-NEXT:    addi 3, 3, 42
# CHECK:         paddi 3, 3, 42, 0
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    lwa 3, 4(3)
# CHECK-NEXT:    blr


	.globl	Plus
	.p2align	4
	.type	Plus,@function
Plus:
.LPlus$local:
.Lpcrel6:
        addi 3, 3, 42
        addi 3, 3, 42
	pld 3, vec@got@pcrel(0), 1
	.reloc .Lpcrel6+8,R_PPC64_PCREL_OPT,.-(.Lpcrel6+8)
	lwa 3, 4(3)
	blr
	.long	0
	.quad	0
# CHECK-LABEL:   Plus
# CHECK:         pld 3, 0(0), 1
# CHECK-NEXT:    R_PPC64_PCREL_OPT	*ABS*+0x8
# CHECK-NEXT:    R_PPC64_GOT_PCREL34	vec
# CHECK-NEXT:    lwa 3, 4(3)
# CHECK-NEXT:    blr

##
# This section of tests contains the variable MCSymbol as part of the
# MCSymbolRefExpr for the first parameter of the .reloc relocation.
##
	.globl	VarLabelMinimal                    # -- Begin function
	.p2align	4
	.type	VarLabelMinimal,@function
VarLabelMinimal:
.LVarLabelMinimal$local:
	pld 3, vec@got@pcrel(0), 1
.Lpcrel101=.-8
	.reloc .Lpcrel101,R_PPC64_PCREL_OPT,.-.Lpcrel101
	lwa 3, 4(3)
	blr
	.long	0
	.quad	0
# CHECK-LABEL:   VarLabelMinimal
# CHECK:         pld 3, 0(0), 1
# CHECK-NEXT:    R_PPC64_GOT_PCREL34	vec
# CHECK-NEXT:    R_PPC64_PCREL_OPT	*ABS*+0x8
# CHECK-NEXT:    lwa 3, 4(3)
# CHECK-NEXT:    blr


	.globl	VarLabelSingleInsnBetween
	.p2align	4
	.type	VarLabelSingleInsnBetween,@function
VarLabelSingleInsnBetween:
.LVarLabelSingleInsnBetween$local:
	pld 3, vec@got@pcrel(0), 1
.Lpcrel102=.-8
        addi 3, 3, 42
	.reloc .Lpcrel102,R_PPC64_PCREL_OPT,.-.Lpcrel102
	lwa 3, 4(3)
	blr
	.long	0
	.quad	0
# CHECK_LABEL:   VarLabelSingleInsnBetween
# CHECK:         pld 3, 0(0), 1
# CHECK-NEXT:    R_PPC64_GOT_PCREL34	vec
# CHECK-NEXT:    R_PPC64_PCREL_OPT	*ABS*+0xc
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    lwa 3, 4(3)
# CHECK-NEXT:    blr

	.globl	VarLabelMultiInsnBetween                    # -- Begin function
	.p2align	4
	.type	VarLabelMultiInsnBetween,@function
VarLabelMultiInsnBetween:
.LVarLabelMultiInsnBetween$local:
	pld 3, vec@got@pcrel(0), 1
.Lpcrel103=.-8
        addi 3, 3, 42
        addi 3, 3, 42
        addi 3, 3, 42
        addi 3, 3, 42
        addi 3, 3, 42
	.reloc .Lpcrel103,R_PPC64_PCREL_OPT,.-.Lpcrel103
	lwa 3, 4(3)
	blr
	.long	0
	.quad	0
# CHECK_LABEL:   VarLabelMultiInsnBetween
# CHECK:         pld 3, 0(0), 1
# CHECK-NEXT:    R_PPC64_GOT_PCREL34	vec
# CHECK-NEXT:    R_PPC64_PCREL_OPT	*ABS*+0x1c
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    lwa 3, 4(3)
# CHECK-NEXT:    blr


	.globl	VarLabelPrefixInsnBetween                    # -- Begin function
	.p2align	4
	.type	VarLabelPrefixInsnBetween,@function
VarLabelPrefixInsnBetween:
.LVarLabelPrefixInsnBetween$local:
	pld 3, vec@got@pcrel(0), 1
.Lpcrel104=.-8
        addi 3, 3, 42
        paddi 3, 3, 42, 0
        addi 3, 3, 42
        paddi 3, 3, 42, 0
        addi 3, 3, 42
	.reloc .Lpcrel104,R_PPC64_PCREL_OPT,.-.Lpcrel104
	lwa 3, 4(3)
	blr
	.long	0
	.quad	0
# CHECK_LABEL:   VarLabelPrefixInsnBetween
# CHECK:         pld 3, 0(0), 1
# CHECK-NEXT:    R_PPC64_GOT_PCREL34	vec
# CHECK-NEXT:    R_PPC64_PCREL_OPT	*ABS*+0x24
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    paddi 3, 3, 42, 0
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    paddi 3, 3, 42, 0
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    lwa 3, 4(3)
# CHECK-NEXT:    blr


	.globl	VarLabelSpaceBetween                    # -- Begin function
	.p2align	4
	.type	VarLabelSpaceBetween,@function
VarLabelSpaceBetween:
.LVarLabelSpaceBetween$local:
	pld 3, vec@got@pcrel(0), 1
.Lpcrel105=.-8
        addi 3, 3, 42
        paddi 3, 3, 42, 0
        addi 3, 3, 42
        .space 40, 0
        paddi 3, 3, 42, 0
        addi 3, 3, 42
	.reloc .Lpcrel105,R_PPC64_PCREL_OPT,.-.Lpcrel105
	lwa 3, 4(3)
	blr
	.long	0
	.quad	0
# CHECK_LABEL:   VarLabelSpaceBetween
# CHECK:         pld 3, 0(0), 1
# CHECK-NEXT:    R_PPC64_GOT_PCREL34	vec
# CHECK-NEXT:    R_PPC64_PCREL_OPT	*ABS*+0x4c
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    paddi 3, 3, 42, 0
# CHECK-NEXT:    addi 3, 3, 42
# CHECK:         paddi 3, 3, 42, 0
# CHECK-NEXT:    addi 3, 3, 42
# CHECK-NEXT:    lwa 3, 4(3)
# CHECK-NEXT:    blr


	.globl	VarLabelPlus
	.p2align	4
	.type	VarLabelPlus,@function
VarLabelPlus:
.LVarLabelPlus$local:
.Lpcrel106:
        addi 3, 3, 42
        addi 3, 3, 42
	pld 3, vec@got@pcrel(0), 1
	.reloc .Lpcrel106+8,R_PPC64_PCREL_OPT,.-(.Lpcrel106+8)
	lwa 3, 4(3)
	blr
	.long	0
	.quad	0
# CHECK-LABEL:   VarLabelPlus
# CHECK:         pld 3, 0(0), 1
# CHECK-NEXT:    R_PPC64_PCREL_OPT	*ABS*+0x8
# CHECK-NEXT:    R_PPC64_GOT_PCREL34	vec
# CHECK-NEXT:    lwa 3, 4(3)
# CHECK-NEXT:    blr

# Check the situation where the PLD requires an alignment nop.
	.globl	AlignPLD
	.p2align	6
	.type	AlignPLD,@function
        .space          60      # Force the pld to require an alignment nop.
AlignPLD:
.LAlignPLD$local:
	pld 3, vec@got@pcrel(0), 1
.Lpcrel201:
	.reloc .Lpcrel201-8,R_PPC64_PCREL_OPT,.-(.Lpcrel201-8)
	lwa 3, 4(3)
	blr
# CHECK-LABEL:   AlignPLD
# CHECK:         nop
# CHECK-NEXT:    pld 3, 0(0), 1
# CHECK-NEXT:    R_PPC64_GOT_PCREL34	vec
# CHECK-NEXT:    R_PPC64_PCREL_OPT	*ABS*+0x8
# CHECK-NEXT:    lwa 3, 4(3)
# CHECK-NEXT:    blr

# The label and the pld are on the same line and so the nop is inserted before
# the label and the relocation should work.
	.globl	AlignPLDSameLine
	.p2align	6
	.type	AlignPLDSameLine,@function
        .space          60      # Force the pld to require an alignment nop.
AlignPLDSameLine:
.LAlignPLDSameLine$local:
.Lpcrel202: pld 3, vec@got@pcrel(0), 1
	.reloc .Lpcrel202,R_PPC64_PCREL_OPT,.-.Lpcrel202
	lwa 3, 4(3)
	blr
# CHECK-LABEL:   AlignPLDSameLine
# CHECK:         nop
# CHECK-NEXT:    pld 3, 0(0), 1
# CHECK-NEXT:    R_PPC64_GOT_PCREL34	vec
# CHECK-NEXT:    R_PPC64_PCREL_OPT	*ABS*+0x8
# CHECK-NEXT:    lwa 3, 4(3)
# CHECK-NEXT:    blr

	.globl	AlignPLDLabelBefore
	.p2align	6
	.type	AlignPLDLabelBefore,@function
        .space          60      # Force the pld to require an alignment nop.
AlignPLDLabelBefore:
.LAlignPLDLabelBefore$local:
.Label:
	pld 3, vec@got@pcrel(0), 1
.Lpcrel203:
	.reloc .Lpcrel203-8,R_PPC64_PCREL_OPT,.-(.Lpcrel203-8)
	lwa 3, 4(3)
	blr
# CHECK-LABEL:   AlignPLDLabelBefore
# CHECK:         nop
# CHECK-NEXT:    pld 3, 0(0), 1
# CHECK-NEXT:    R_PPC64_GOT_PCREL34	vec
# CHECK-NEXT:    R_PPC64_PCREL_OPT	*ABS*+0x8
# CHECK-NEXT:    lwa 3, 4(3)
# CHECK-NEXT:    blr

	.globl	AlignPLDLabelSameLine
	.p2align	6
	.type	AlignPLDLabelSameLine,@function
        .space          60      # Force the pld to require an alignment nop.
AlignPLDLabelSameLine:
.Label2: pld 3, vec@got@pcrel(0), 1
.Lpcrel204:
	.reloc .Lpcrel204-8,R_PPC64_PCREL_OPT,.-(.Lpcrel204-8)
	lwa 3, 4(3)
	blr
# CHECK-LABEL:   AlignPLDLabelSameLine
# CHECK:         nop
# CHECK-NEXT:    pld 3, 0(0), 1
# CHECK-NEXT:    R_PPC64_GOT_PCREL34	vec
# CHECK-NEXT:    R_PPC64_PCREL_OPT	*ABS*+0x8
# CHECK-NEXT:    lwa 3, 4(3)
# CHECK-NEXT:    blr
