# RUN: llvm-mc -triple i386-unknown-unknown %s -filetype obj -o - \
# RUN:   | llvm-readobj --symbols | FileCheck %s

	.end

its_a_tarp:
	int $0x3

# CHECK: Symbol {
# CHECK-NOT:   Name: its_a_tarp

