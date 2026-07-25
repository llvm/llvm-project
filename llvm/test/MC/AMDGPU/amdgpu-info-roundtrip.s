// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=asm %s | FileCheck --check-prefix=ASM %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %s | llvm-readobj -r --sections --section-data --string-dump=.amdgpu.strtab - | FileCheck --check-prefix=OBJ %s

// Test that .amdgpu_info directives round-trip through the assembler (asm and
// object emission) and produce the correct TLV-encoded .amdgpu.info section.

	.text
	.globl	my_kernel
	.p2align	8
	.type	my_kernel,@function
my_kernel:
	s_endpgm
.Lfunc_end0:
	.size	my_kernel, .Lfunc_end0-my_kernel

	.globl	helper
	.p2align	2
	.type	helper,@function
helper:
	s_setpc_b64 s[30:31]
.Lfunc_end1:
	.size	helper, .Lfunc_end1-helper

	.globl	addr_taken_func
	.p2align	2
	.type	addr_taken_func,@function
addr_taken_func:
	s_setpc_b64 s[30:31]
.Lfunc_end2:
	.size	addr_taken_func, .Lfunc_end2-addr_taken_func

	.globl	extern_func

// COM: Kernel: flags=7 (KERNEL|VCC|FLAT_SCRATCH), resources, call edge, use
// COM: edge, indirect call, and type ID. Non-zero AGPR to verify conditional
// COM: emission.
	.amdgpu_info my_kernel
		.amdgpu_flags 7
		.amdgpu_num_sgpr 33
		.amdgpu_num_vgpr 32
		.amdgpu_num_agpr 4
		.amdgpu_private_segment_size 0
		.amdgpu_use lds_var
		.amdgpu_call helper
		.amdgpu_indirect_call "vi"
	.end_amdgpu_info

// COM: Device function: flags=2 (VCC), call edge to external. Zero AGPR values
// COM: are omitted from the input; the parser defaults them to 0 and the
// COM: emitter skips them.
	.amdgpu_info helper
		.amdgpu_flags 2
		.amdgpu_num_sgpr 8
		.amdgpu_num_vgpr 10
		.amdgpu_private_segment_size 16
		.amdgpu_call extern_func
	.end_amdgpu_info

// Address-taken function with type ID. Zero AGPR omitted.
	.amdgpu_info addr_taken_func
		.amdgpu_flags 0
		.amdgpu_num_sgpr 2
		.amdgpu_num_vgpr 4
		.amdgpu_private_segment_size 0
		.amdgpu_typeid "vi"
	.end_amdgpu_info

// ASM: .amdgpu_info my_kernel
// ASM: .amdgpu_flags 7
// ASM: .amdgpu_num_sgpr 33
// ASM: .amdgpu_num_vgpr 32
// ASM: .amdgpu_num_agpr 4
// ASM: .amdgpu_private_segment_size 0
// ASM: .amdgpu_use lds_var
// ASM: .amdgpu_call helper
// ASM: .amdgpu_indirect_call "vi"
// ASM: .end_amdgpu_info

// ASM: .amdgpu_info helper
// ASM: .amdgpu_flags 2
// ASM: .amdgpu_num_sgpr 8
// ASM: .amdgpu_num_vgpr 10
// ASM-NOT: .amdgpu_num_agpr
// ASM: .amdgpu_private_segment_size 16
// ASM: .amdgpu_call extern_func
// ASM: .end_amdgpu_info

// ASM: .amdgpu_info addr_taken_func
// ASM: .amdgpu_flags 0
// ASM: .amdgpu_num_sgpr 2
// ASM: .amdgpu_num_vgpr 4
// ASM-NOT: .amdgpu_num_agpr
// ASM: .amdgpu_private_segment_size 0
// ASM: .amdgpu_typeid "vi"
// ASM: .end_amdgpu_info

// OBJ: Section {
// OBJ:   Name: .amdgpu.info
// OBJ:   Type: SHT_PROGBITS
// OBJ:   Flags [
// OBJ:     SHF_EXCLUDE
// OBJ:   ]
// OBJ: }

// The string pool backs INFO_INDIRECT_CALL / INFO_TYPEID payloads. It is an
// ELF-convention SHT_STRTAB with a leading null byte at offset 0 and string
// deduplication -- both directives above reference the same "vi" TypeID, so
// it must appear exactly once starting at offset 1.
// OBJ: Section {
// OBJ:   Name: .amdgpu.strtab
// OBJ:   Type: SHT_STRTAB
// OBJ:   Flags [
// OBJ:     SHF_EXCLUDE
// OBJ:   ]
// OBJ: }

// Relocations in .amdgpu.info should reference defined and external symbols.
// OBJ-DAG: R_AMDGPU_ABS64 my_kernel
// OBJ-DAG: R_AMDGPU_ABS64 helper
// OBJ-DAG: R_AMDGPU_ABS64 addr_taken_func
// OBJ-DAG: R_AMDGPU_ABS64 extern_func
// OBJ-DAG: R_AMDGPU_ABS64 lds_var

// OBJ: String dump of section '.amdgpu.strtab':
// OBJ-NEXT: [{{ +}}1] vi
// OBJ-NOT:  ] vi
