// RUN: llvm-mc -filetype=obj -triple arm-eabi %s -o - | llvm-readobj -s -t | FileCheck %s
// Test that global variables and functions are assigned correct section.
	.text
	.syntax unified
	.eabi_attribute	67, "2.09"	@ Tag_conformance
	.eabi_attribute	6, 1	@ Tag_CPU_arch
	.eabi_attribute	8, 1	@ Tag_ARM_ISA_use
	.eabi_attribute	17, 1	@ Tag_ABI_PCS_GOT_use
	.eabi_attribute	20, 1	@ Tag_ABI_FP_denormal
	.eabi_attribute	21, 1	@ Tag_ABI_FP_exceptions
	.eabi_attribute	23, 3	@ Tag_ABI_FP_number_model
	.eabi_attribute	34, 1	@ Tag_CPU_unaligned_access
	.eabi_attribute	24, 1	@ Tag_ABI_align_needed
	.eabi_attribute	25, 1	@ Tag_ABI_align_preserved
	.eabi_attribute	38, 1	@ Tag_ABI_FP_16bit_format
	.eabi_attribute	18, 4	@ Tag_ABI_PCS_wchar_t
	.eabi_attribute	26, 2	@ Tag_ABI_enum_size
	.eabi_attribute	14, 0	@ Tag_ABI_PCS_R9_use
	.section	my_text.1,"ax",%progbits
	.globl	foo
	.p2align	2
	.type	foo,%function
	.code	32                      @ @foo
foo:
	.fnstart
@ BB#0:                                 @ %entry
	ldr	r0, .LCPI0_0
	ldr	r0, [r0]
	mov	pc, lr
	.p2align	2
@ BB#1:
.LCPI0_0:
	.long	b
.Lfunc_end0:
	.size	foo, .Lfunc_end0-foo
	.cantunwind
	.fnend

	.section	my_text.2,"ax",%progbits
	.globl	goo
	.p2align	2
	.type	goo,%function
	.code	32                      @ @goo
goo:
	.fnstart
@ BB#0:                                 @ %entry
	.save	{r11, lr}
	push	{r11, lr}
	ldr	r0, .LCPI1_0
	ldr	r1, .LCPI1_1
	bl	zoo
	pop	{r11, lr}
	mov	pc, lr
	.p2align	2
@ BB#1:
.LCPI1_0:
	.long	_ZL1g
.LCPI1_1:
	.long	_ZZ3gooE7lstat_h
.Lfunc_end1:
	.size	goo, .Lfunc_end1-goo
	.cantunwind
	.fnend

	.text
	.globl	hoo
	.p2align	2
	.type	hoo,%function
	.code	32                      @ @hoo
hoo:
	.fnstart
@ BB#0:                                 @ %entry
	ldr	r0, .LCPI2_0
	ldr	r0, [r0]
	mov	pc, lr
	.p2align	2
@ BB#1:
.LCPI2_0:
	.long	b
.Lfunc_end2:
	.size	hoo, .Lfunc_end2-hoo
	.cantunwind
	.fnend

	.type	a,%object               @ @a
	.section	my_bss.1,"aw",%nobits
	.globl	a
	.p2align	2
a:
	.long	0                       @ 0x0
	.size	a, 4

	.type	b,%object               @ @b
	.section	my_data.1,"aw",%progbits
	.globl	b
	.p2align	2
b:
	.long	1                       @ 0x1
	.size	b, 4

	.type	c,%object               @ @c
	.section	my_bss.1,"aw",%nobits
	.globl	c
	.p2align	2
c:
	.zero	16
	.size	c, 16

	.type	d,%object               @ @d
	.globl	d
	.p2align	1
d:
	.zero	10
	.size	d, 10

	.type	e,%object               @ @e
	.section	my_data.1,"aw",%progbits
	.globl	e
	.p2align	1
e:
	.short	0                       @ 0x0
	.short	0                       @ 0x0
	.short	1                       @ 0x1
	.short	0                       @ 0x0
	.short	0                       @ 0x0
	.short	0                       @ 0x0
	.size	e, 12

	.type	f,%object               @ @f
	.section	my_rodata.1,"a",%progbits
	.globl	f
	.p2align	2
f:
	.long	2                       @ 0x2
	.size	f, 4

	.type	h,%object               @ @h
	.bss
	.globl	h
	.p2align	2
h:
	.long	0                       @ 0x0
	.size	h, 4

	.type	i,%object               @ @i
	.section	my_bss.2,"aw",%nobits
	.globl	i
	.p2align	2
i:
	.long	0                       @ 0x0
	.size	i, 4

	.type	j,%object               @ @j
	.section	my_rodata.1,"a",%progbits
	.globl	j
	.p2align	2
j:
	.long	4                       @ 0x4
	.size	j, 4

	.type	k,%object               @ @k
	.section	my_bss.2,"aw",%nobits
	.globl	k
	.p2align	2
k:
	.long	0                       @ 0x0
	.size	k, 4

	.type	_ZZ3gooE7lstat_h,%object @ @_ZZ3gooE7lstat_h
	.p2align	2
_ZZ3gooE7lstat_h:
	.long	0                       @ 0x0
	.size	_ZZ3gooE7lstat_h, 4

	.type	_ZL1g,%object           @ @_ZL1g
	.section	my_bss.1,"aw",%nobits
	.p2align	2
_ZL1g:
	.zero	8
	.size	_ZL1g, 8

	.type	l,%object               @ @l
	.section	my_data.2,"aw",%progbits
	.globl	l
	.p2align	2
l:
	.long	5                       @ 0x5
	.size	l, 4

	.type	m,%object               @ @m
	.section	my_rodata.2,"a",%progbits
	.globl	m
	.p2align	2
m:
	.long	6                       @ 0x6
	.size	m, 4

	.type	n,%object               @ @n
	.bss
	.globl	n
	.p2align	2
n:
	.long	0                       @ 0x0
	.size	n, 4

	.type	o,%object               @ @o
	.data
	.globl	o
	.p2align	2
o:
	.long	6                       @ 0x6
	.size	o, 4

	.type	p,%object               @ @p
	.section	.rodata,"a",%progbits
	.globl	p
	.p2align	2
p:
	.long	7                       @ 0x7
	.size	p, 4


	.ident	"clang version 5.0.0 (http://llvm.org/git/clang.git 254242a3ad440307fb451093a429c71ea9a8c888) (http://llvm.org/git/llvm.git 3c8daefbe3d1672ac1dae775b211f881f0063038)"
	.section	".note.GNU-stack","",%progbits
	.eabi_attribute	30, 1	@ Tag_ABI_optimization_goals

//CHECK:   Section {
//CHECK:     Name: .text
//CHECK:     Type: SHT_PROGBITS (0x1)
//CHECK:     Flags [ (0x6)
//CHECK:       SHF_ALLOC (0x2)
//CHECK:       SHF_EXECINSTR (0x4)
//CHECK:     ]
//CHECK:   }
//CHECK:   Section {
//CHECK:     Name: my_text.1
//CHECK:     Type: SHT_PROGBITS (0x1)
//CHECK:     Flags [ (0x6)
//CHECK:       SHF_ALLOC (0x2)
//CHECK:       SHF_EXECINSTR (0x4)
//CHECK:     ]
//CHECK:   }
//CHECK:   Section {
//CHECK:     Name: my_text.2
//CHECK:     Type: SHT_PROGBITS (0x1)
//CHECK:     Flags [ (0x6)
//CHECK:       SHF_ALLOC (0x2)
//CHECK:       SHF_EXECINSTR (0x4)
//CHECK:     ]
//CHECK:   }
//CHECK:   Section {
//CHECK:     Name: my_bss.1
//CHECK:     Type: SHT_NOBITS (0x8)
//CHECK:     Flags [ (0x3)
//CHECK:       SHF_ALLOC (0x2)
//CHECK:       SHF_WRITE (0x1)
//CHECK:     ]
//CHECK:   }
//CHECK:   Section {
//CHECK:     Name: my_data.1
//CHECK:     Type: SHT_PROGBITS (0x1)
//CHECK:     Flags [ (0x3)
//CHECK:       SHF_ALLOC (0x2)
//CHECK:       SHF_WRITE (0x1)
//CHECK:     ]
//CHECK:   }
//CHECK:   Section {
//CHECK:     Name: my_rodata.1
//CHECK:     Type: SHT_PROGBITS (0x1)
//CHECK:     Flags [ (0x2)
//CHECK:       SHF_ALLOC (0x2)
//CHECK:     ]
//CHECK:   }
//CHECK:   Section {
//CHECK:     Name: .bss
//CHECK:     Type: SHT_NOBITS (0x8)
//CHECK:     Flags [ (0x3)
//CHECK:       SHF_ALLOC (0x2)
//CHECK:       SHF_WRITE (0x1)
//CHECK:     ]
//CHECK:   }
//CHECK:   Section {
//CHECK:     Name: my_bss.2
//CHECK:     Type: SHT_NOBITS (0x8)
//CHECK:     Flags [ (0x3)
//CHECK:       SHF_ALLOC (0x2)
//CHECK:       SHF_WRITE (0x1)
//CHECK:     ]
//CHECK:   }
//CHECK:   Section {
//CHECK:     Name: my_data.2
//CHECK:     Type: SHT_PROGBITS (0x1)
//CHECK:     Flags [ (0x3)
//CHECK:       SHF_ALLOC (0x2)
//CHECK:       SHF_WRITE (0x1)
//CHECK:     ]
//CHECK:   }
//CHECK:   Section {
//CHECK:     Name: my_rodata.2
//CHECK:     Type: SHT_PROGBITS (0x1)
//CHECK:     Flags [ (0x2)
//CHECK:       SHF_ALLOC (0x2)
//CHECK:     ]
//CHECK:   }
//CHECK:   Section {
//CHECK:     Name: .data
//CHECK:     Type: SHT_PROGBITS (0x1)
//CHECK:     Flags [ (0x3)
//CHECK:       SHF_ALLOC (0x2)
//CHECK:       SHF_WRITE (0x1)
//CHECK:     ]
//CHECK:   }
//CHECK:   Section {
//CHECK:     Name: .rodata
//CHECK:     Type: SHT_PROGBITS (0x1)
//CHECK:     Flags [ (0x2)
//CHECK:       SHF_ALLOC (0x2)
//CHECK:     ]
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: _ZL1g
//CHECK:     Section: my_bss.1 (0xE)
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: _ZZ3gooE7lstat_h
//CHECK:     Section: my_bss.2 (0x12)
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: a
//CHECK:     Section: my_bss.1 (0xE)
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: b
//CHECK:     Section: my_data.1 (0xF)
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: c
//CHECK:     Section: my_bss.1 (0xE)
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: d
//CHECK:     Section: my_bss.1 (0xE)
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: e
//CHECK:     Section: my_data.1 (0xF)
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: f
//CHECK:     Section: my_rodata.1 (0x10)
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: foo
//CHECK:     Section: my_text.1 (0x4)
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: goo
//CHECK:     Section: my_text.2 (0x8)
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: h
//CHECK:     Section: .bss (0x11)
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: hoo
//CHECK:     Section: .text (0x2)
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: i
//CHECK:     Section: my_bss.2 (0x12)
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: j
//CHECK:     Section: my_rodata.1 (0x10)
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: k
//CHECK:     Section: my_bss.2 (0x12)
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: l
//CHECK:     Section: my_data.2 (0x13)
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: m
//CHECK:     Section: my_rodata.2 (0x14)
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: n
//CHECK:     Section: .bss (0x11)
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: o
//CHECK:     Section: .data (0x15)
//CHECK:   }
//CHECK:   Symbol {
//CHECK:     Name: p
//CHECK:     Section: .rodata (0x16)
//CHECK:   }
