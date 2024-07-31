// RUN: llvm-mc -filetype=obj -triple i386 %s -o - | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

// This is a case where llvm-mc computes a better layout than Darwin 'as'. This
// issue is that after the first jmp slides, the .align size must be
// recomputed -- otherwise the second jump will appear to be out-of-range for a
// 1-byte jump.

// CHECK:            int3
// CHECK-NEXT:  ce:  int3
// CHECK:       d0:  pushal
// CHECK:      130:  jl      0xd0

L0:
        .space 0x8a, 0x90
	jmp	L0
        .space (0xb3 - 0x8f), 0x90
	jle	L2
        .space (0xcd - 0xb5), 0x90
	.p2align 4, 0xcc
L1:
        .space (0x130 - 0xd0),0x60
	jl	L1
L2:
