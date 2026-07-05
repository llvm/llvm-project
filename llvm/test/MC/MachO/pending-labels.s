// Verify relocations for temporary labels are referenced by real symbols
// at the same address.
//
// RUN: llvm-mc -triple x86_64-apple-darwin -filetype=obj -o - %s | llvm-objdump -r - | FileCheck %s

L1:
	.section __TEXT,__text_cold,regular,pure_instructions
L2:
	.text
L3:
	.section __TEXT,__text_cold,regular,pure_instructions
L4:
_function2:
L5:
	nop
L6:
	.text
L7:
_function1:
L8:
	nop

	.data
__data:
	.quad L1-.
	.quad L2-.
	.quad L3-.
	.quad L4-.
	.quad L5-.
	.quad L6-.
	.quad L7-.
	.quad L8-.
// CHECK: 0000000000000038 X86_64_RELOC_SUBTRACTOR _function1-__data
// CHECK: 0000000000000038 X86_64_RELOC_UNSIGNED _function1
// CHECK: 0000000000000030 X86_64_RELOC_SUBTRACTOR __text-__data
// CHECK: 0000000000000030 X86_64_RELOC_UNSIGNED __text
// CHECK: 0000000000000028 X86_64_RELOC_SUBTRACTOR _function2-__data
// CHECK: 0000000000000028 X86_64_RELOC_UNSIGNED _function2
// CHECK: 0000000000000020 X86_64_RELOC_SUBTRACTOR _function2-__data
// CHECK: 0000000000000020 X86_64_RELOC_UNSIGNED _function2
// CHECK: 0000000000000018 X86_64_RELOC_SUBTRACTOR __text_cold-__data
// CHECK: 0000000000000018 X86_64_RELOC_UNSIGNED __text_cold
// CHECK: 0000000000000010 X86_64_RELOC_SUBTRACTOR __text-__data
// CHECK: 0000000000000010 X86_64_RELOC_UNSIGNED __text
// CHECK: 0000000000000008 X86_64_RELOC_SUBTRACTOR __text_cold-__data
// CHECK: 0000000000000008 X86_64_RELOC_UNSIGNED __text_cold
// CHECK: 0000000000000000 X86_64_RELOC_SUBTRACTOR __text-__data
// CHECK: 0000000000000000 X86_64_RELOC_UNSIGNED __text
