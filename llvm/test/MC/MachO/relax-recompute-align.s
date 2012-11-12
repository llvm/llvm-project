// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

// FIXME: This is a horrible way of checking the output, we need an llvm-mc
// based 'otool'.

// This is a case where llvm-mc computes a better layout than Darwin 'as'. This
// issue is that after the first jmp slides, the .align size must be
// recomputed -- otherwise the second jump will appear to be out-of-range for a
// 1-byte jump.

// CHECK:  # Section 0
// CHECK: (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:  ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:  ('address', 0)
// CHECK:  ('size', 306)
// CHECK:  ('offset', 324)
// CHECK:  ('alignment', 4)
// CHECK:  ('reloc_offset', 0)
// CHECK:  ('num_reloc', 0)
// CHECK:  ('flags', 0x80000400)
// CHECK:  ('reserved1', 0)
// CHECK:  ('reserved2', 0)
// CHECK: ),

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
