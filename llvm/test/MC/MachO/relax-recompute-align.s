// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj -S - | FileCheck %s

// FIXME: This is a horrible way of checking the output, we need an llvm-mc
// based 'otool'.

// This is a case where llvm-mc computes a better layout than Darwin 'as'. This
// issue is that after the first jmp slides, the .align size must be
// recomputed -- otherwise the second jump will appear to be out-of-range for a
// 1-byte jump.

L0:
        .space 0x8a, 0x90
	jmp	L0
        .space (0xb3 - 0x8f), 0x90
	jle	L2
        .space (0xcd - 0xb5), 0x90
	.align	4, 0x90
L1:
        .space (0x130 - 0xd0),0x90
	jl	L1
L2:

.zerofill __DATA,__bss,_sym,4,2

// CHECK: Section {
// CHECK-NEXT:   Index: 0
// CHECK-NEXT:   Name: __text (5F 5F 74 65 78 74 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:   Segment: __TEXT (5F 5F 54 45 58 54 00 00 00 00 00 00 00 00 00 00)
// CHECK-NEXT:   Address: 0x0
// CHECK-NEXT:   Size: 0x132
// CHECK-NEXT:   Offset: 340
// CHECK-NEXT:   Alignment: 4
// CHECK-NEXT:   RelocationOffset: 0x0
// CHECK-NEXT:   RelocationCount: 0
// CHECK-NEXT:   Type: Regular (0x0)
// CHECK-NEXT:   Attributes [ (0x800004)
// CHECK-NEXT:     PureInstructions (0x800000)
// CHECK-NEXT:     SomeInstructions (0x4)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Reserved1: 0x0
// CHECK-NEXT:   Reserved2: 0x0
// CHECK-NEXT: }
