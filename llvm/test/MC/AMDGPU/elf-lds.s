// RUN: llvm-mc -filetype=obj -triple amdgcn-- -mcpu gfx900 %s -o - | llvm-readobj -r --syms - | FileCheck %s

	.text
	.globl test_kernel
	.p2align 8
	.type test_kernel,@function
test_kernel:
	s_mov_b32 s0, lds0@abs32@lo
	v_lshl_add_u32 v3, v0, 2, s0
	ds_read2_b32 v[1:2], v3 offset1:1

	s_mov_b32 s0, lds4@abs32@lo
	v_lshl_add_u32 v3, v0, 2, s0
	ds_write_b32 v3, v1
	s_endpgm
.Lfunc_end:
	.size test_kernel, .Lfunc_end-test_kernel

	.globl lds0
	.amdgpu_lds lds0, 192, 16

	.globl lds1
	.amdgpu_lds lds1,387,8

	; Weird whitespace cases
	.globl lds2
	.amdgpu_lds lds2, 12

	; No alignment or .globl directive, not mentioned anywhere
	.amdgpu_lds lds3, 16

	; No alignment or .globl directive, size 0, but mentioned in .text
	.amdgpu_lds lds4, 0

// CHECK:      Relocations [
// CHECK:        Section (3) .rel.text {
// CHECK-NEXT:     0x4 R_AMDGPU_ABS32 lds0
// CHECK-NEXT:     0x1C R_AMDGPU_ABS32 lds4
// CHECK-NEXT:   }
// CHECK:      ]

// CHECK:      Symbol {
// CHECK:        Name: lds0 (54)
// CHECK-NEXT:   Value: 0x10
// CHECK-NEXT:   Size: 192
// CHECK-NEXT:   Binding: Global (0x1)
// CHECK-NEXT:   Type: Object (0x1)
// CHECK-NEXT:   Other: 0
// CHECK-NEXT:   Section: Processor Specific (0xFF00)
// CHECK-NEXT: }

// CHECK:      Symbol {
// CHECK:        Name: lds4 (39)
// CHECK-NEXT:   Value: 0x4
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Binding: Global (0x1)
// CHECK-NEXT:   Type: Object (0x1)
// CHECK-NEXT:   Other: 0
// CHECK-NEXT:   Section: Processor Specific (0xFF00)
// CHECK-NEXT: }

// CHECK:      Symbol {
// CHECK:        Name: lds1 (49)
// CHECK-NEXT:   Value: 0x8
// CHECK-NEXT:   Size: 387
// CHECK-NEXT:   Binding: Global (0x1)
// CHECK-NEXT:   Type: Object (0x1)
// CHECK-NEXT:   Other: 0
// CHECK-NEXT:   Section: Processor Specific (0xFF00)
// CHECK-NEXT: }

// CHECK:      Symbol {
// CHECK:        Name: lds2 (44)
// CHECK-NEXT:   Value: 0x4
// CHECK-NEXT:   Size: 12
// CHECK-NEXT:   Binding: Global (0x1)
// CHECK-NEXT:   Type: Object (0x1)
// CHECK-NEXT:   Other: 0
// CHECK-NEXT:   Section: Processor Specific (0xFF00)
// CHECK-NEXT: }

// CHECK-NOT:    Name: lds3
