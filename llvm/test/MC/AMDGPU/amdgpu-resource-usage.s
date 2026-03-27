// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck --check-prefix=ASM %s
// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj < %s > %t
// RUN: llvm-readelf -S %t | FileCheck --check-prefix=ELF-SEC %s
// RUN: llvm-readelf -r %t | FileCheck --check-prefix=ELF-RELOC %s

.text

.globl bar
.type bar,@function
bar:
  s_endpgm

.type baz,@function
baz:
  s_endpgm

.globl foo
.type foo,@function
foo:
  s_endpgm

.type quux,@function
quux:
  s_endpgm

.extern external_fn

// ASM: .amdgpu_resource_usage bar
// ASM-NEXT: .num_vgpr 65
// ASM-NEXT: .num_agpr 0
// ASM-NEXT: .num_sgpr 25
// ASM-NEXT: .named_barrier 0
// ASM-NEXT: .private_seg_size 16
// ASM-NEXT: .uses_vcc 1
// ASM-NEXT: .uses_flat_scratch 0
// ASM-NEXT: .has_dyn_sized_stack 0
// ASM-NEXT: .end_amdgpu_resource_usage
	.amdgpu_resource_usage bar
		.num_vgpr 65
		.num_agpr 0
		.num_sgpr 25
		.named_barrier 0
		.private_seg_size 16
		.uses_vcc 1
		.uses_flat_scratch 0
		.has_dyn_sized_stack 0
	.end_amdgpu_resource_usage

// ASM: .amdgpu_resource_usage foo
// ASM-NEXT: .num_vgpr 10
// ASM-NEXT: .num_agpr 4
// ASM-NEXT: .num_sgpr 8
// ASM-NEXT: .named_barrier 2
// ASM-NEXT: .private_seg_size 0
// ASM-NEXT: .uses_vcc 0
// ASM-NEXT: .uses_flat_scratch 1
// ASM-NEXT: .has_dyn_sized_stack 1
// ASM-NEXT: .end_amdgpu_resource_usage
	.amdgpu_resource_usage foo
		.num_vgpr 10
		.num_agpr 4
		.num_sgpr 8
		.named_barrier 2
		.private_seg_size 0
		.uses_vcc 0
		.uses_flat_scratch 1
		.has_dyn_sized_stack 1
	.end_amdgpu_resource_usage

// ASM: .amdgpu_resource_usage baz
// ASM-NEXT: .num_vgpr 2
// ASM-NEXT: .num_agpr 0
// ASM-NEXT: .num_sgpr 4
// ASM-NEXT: .named_barrier 0
// ASM-NEXT: .private_seg_size 0
// ASM-NEXT: .uses_vcc 0
// ASM-NEXT: .uses_flat_scratch 0
// ASM-NEXT: .has_dyn_sized_stack 0
// ASM-NEXT: .end_amdgpu_resource_usage
	.amdgpu_resource_usage baz
		.num_vgpr 2
		.num_agpr 0
		.num_sgpr 4
		.named_barrier 0
		.private_seg_size 0
		.uses_vcc 0
		.uses_flat_scratch 0
		.has_dyn_sized_stack 0
	.end_amdgpu_resource_usage

// ASM-NOT: .amdgpu_resource_usage quux

// ELF-SEC: .AMDGPU.resource_usage PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} 000060 18 E 0 0 1

// ELF-RELOC:      Relocation section '.rela.AMDGPU.resource_usage'
// ELF-RELOC:      0000000000000000 {{[0-9a-f]+}} R_AMDGPU_NONE {{[0-9a-f]+}} bar + 0
// ELF-RELOC-NEXT: 0000000000000018 {{[0-9a-f]+}} R_AMDGPU_NONE {{[0-9a-f]+}} foo + 0
// ELF-RELOC-NEXT: 0000000000000030 {{[0-9a-f]+}} R_AMDGPU_NONE {{[0-9a-f]+}} baz + 0
// ELF-RELOC-NEXT: 0000000000000048 {{[0-9a-f]+}} R_AMDGPU_NONE {{[0-9a-f]+}} quux + 0

