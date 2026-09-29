// #check radix formats of llvm-nm
// RUN: llvm-mc %s -filetype=obj -triple=x86_64-pc-linux -o %t.o
// RUN: llvm-nm --radix=d %t.o | FileCheck %s
// RUN: llvm-nm --radix=o %t.o | FileCheck --check-prefix=OCTAL %s
// RUN: llvm-nm -tx %t.o | FileCheck --check-prefix=HEX %s
// RUN: llvm-nm -t x %t.o | FileCheck --check-prefix=HEX %s

	.text
	.file	"1.c"
	.type	i0,@object              # @i0
	.bss
	.globl	i0
	.align	4
i0:
	.long	0                       # 0x0
	.size	i0, 4

	.type	i1,@object              # @i1
	.data
	.globl	i1
	.align	4
i1:
	.long	1                       # 0x1
	.size	i1, 4

	.type	i2,@object              # @i2
	.globl	i2
	.align	4
i2:
	.long	2                       # 0x2
	.size	i2, 4

	.type	i3,@object              # @i3
	.globl	i3
	.align	4
i3:
	.long	3                       # 0x3
	.size	i3, 4

	.type	i4,@object              # @i4
	.globl	i4
	.align	4
i4:
	.long	4                       # 0x4
	.size	i4, 4

	.type	i5,@object              # @i5
	.globl	i5
	.align	4
i5:
	.long	5                       # 0x5
	.size	i5, 4

	.type	i6,@object              # @i6
	.globl	i6
	.align	4
i6:
	.long	6                       # 0x6
	.size	i6, 4

	.type	i7,@object              # @i7
	.globl	i7
	.align	4
i7:
	.long	7                       # 0x7
	.size	i7, 4

	.type	i8,@object              # @i8
	.globl	i8
	.align	4
i8:
	.long	8                       # 0x8
	.size	i8, 4

	.type	i9,@object              # @i9
	.globl	i9
	.align	4
i9:
	.long	9                       # 0x9
	.size	i9, 4

	.type	i10,@object             # @i10
	.globl	i10
	.align	4
i10:
	.long	10                      # 0xa
	.size	i10, 4

	.type	i11,@object             # @i11
	.globl	i11
	.align	4
i11:
	.long	11                      # 0xb
	.size	i11, 4

	.type	i12,@object             # @i12
	.globl	i12
	.align	4
i12:
	.long	12                      # 0xc
	.size	i12, 4

	.type	i13,@object             # @i13
	.globl	i13
	.align	4
i13:
	.long	13                      # 0xd
	.size	i13, 4

	.type	i14,@object             # @i14
	.globl	i14
	.align	4
i14:
	.long	14                      # 0xe
	.size	i14, 4

	.type	i15,@object             # @i15
	.globl	i15
	.align	4
i15:
	.long	15                      # 0xf
	.size	i15, 4

	.type	i16,@object             # @i16
	.globl	i16
	.align	4
i16:
	.long	16                      # 0x10
	.size	i16, 4

	.type	i17,@object             # @i17
	.globl	i17
	.align	4
i17:
	.long	17                      # 0x11
	.size	i17, 4

	.type	i18,@object             # @i18
	.globl	i18
	.align	4
i18:
	.long	18                      # 0x12
	.size	i18, 4

	.type	i19,@object             # @i19
	.globl	i19
	.align	4
i19:
	.long	19                      # 0x13
	.size	i19, 4

	.type	i20,@object             # @i20
	.globl	i20
	.align	4
i20:
	.long	20                      # 0x14
	.size	i20, 4

	.type	i21,@object             # @i21
	.globl	i21
	.align	4
i21:
	.long	21                      # 0x15
	.size	i21, 4

	.type	i22,@object             # @i22
	.globl	i22
	.align	4
i22:
	.long	22                      # 0x16
	.size	i22, 4

	.type	i23,@object             # @i23
	.globl	i23
	.align	4
i23:
	.long	23                      # 0x17
	.size	i23, 4

	.type	i24,@object             # @i24
	.globl	i24
	.align	4
i24:
	.long	24                      # 0x18
	.size	i24, 4


	.ident	"clang version 3.6.0 (tags/RELEASE_360/final)"
	.section	".note.GNU-stack","",@progbits

//CHECK:    0000000000000000 B i0
//CHECK:    0000000000000000 D i1
//CHECK:    0000000000000036 D i10
//CHECK:    0000000000000040 D i11
//CHECK:    0000000000000044 D i12
//CHECK:    0000000000000048 D i13
//CHECK:    0000000000000052 D i14
//CHECK:    0000000000000056 D i15
//CHECK:    0000000000000060 D i16
//CHECK:    0000000000000064 D i17
//CHECK:    0000000000000068 D i18
//CHECK:    0000000000000072 D i19
//CHECK:    0000000000000004 D i2
//CHECK:    0000000000000076 D i20
//CHECK:    0000000000000080 D i21
//CHECK:    0000000000000084 D i22
//CHECK:    0000000000000088 D i23
//CHECK:    0000000000000092 D i24
//CHECK:    0000000000000008 D i3
//CHECK:    0000000000000012 D i4
//CHECK:    0000000000000016 D i5
//CHECK:    0000000000000020 D i6
//CHECK:    0000000000000024 D i7
//CHECK:    0000000000000028 D i8
//CHECK:    0000000000000032 D i9

//OCTAL:    0000000000000000 B i0
//OCTAL:    0000000000000000 D i1
//OCTAL:    0000000000000044 D i10
//OCTAL:    0000000000000050 D i11
//OCTAL:    0000000000000054 D i12
//OCTAL:    0000000000000060 D i13
//OCTAL:    0000000000000064 D i14
//OCTAL:    0000000000000070 D i15
//OCTAL:    0000000000000074 D i16
//OCTAL:    0000000000000100 D i17
//OCTAL:    0000000000000104 D i18
//OCTAL:    0000000000000110 D i19
//OCTAL:    0000000000000004 D i2
//OCTAL:    0000000000000114 D i20
//OCTAL:    0000000000000120 D i21
//OCTAL:    0000000000000124 D i22
//OCTAL:    0000000000000130 D i23
//OCTAL:    0000000000000134 D i24
//OCTAL:    0000000000000010 D i3
//OCTAL:    0000000000000014 D i4
//OCTAL:    0000000000000020 D i5
//OCTAL:    0000000000000024 D i6
//OCTAL:    0000000000000030 D i7
//OCTAL:    0000000000000034 D i8
//OCTAL:    0000000000000040 D i9

//HEX:    0000000000000000 B i0
//HEX:    0000000000000000 D i1
//HEX:    0000000000000024 D i10
//HEX:    0000000000000028 D i11
//HEX:    000000000000002c D i12
//HEX:    0000000000000030 D i13
//HEX:    0000000000000034 D i14
//HEX:    0000000000000038 D i15
//HEX:    000000000000003c D i16
//HEX:    0000000000000040 D i17
//HEX:    0000000000000044 D i18
//HEX:    0000000000000048 D i19
//HEX:    0000000000000004 D i2
//HEX:    000000000000004c D i20
//HEX:    0000000000000050 D i21
//HEX:    0000000000000054 D i22
//HEX:    0000000000000058 D i23
//HEX:    000000000000005c D i24
//HEX:    0000000000000008 D i3
//HEX:    000000000000000c D i4
//HEX:    0000000000000010 D i5
//HEX:    0000000000000014 D i6
//HEX:    0000000000000018 D i7
//HEX:    000000000000001c D i8
//HEX:    0000000000000020 D i9
