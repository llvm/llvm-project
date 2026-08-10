# RUN: llvm-mc -triple powerpc64-unknown-linux-gnu < %s | FileCheck %s
# RUN: llvm-mc -triple powerpc64le-unknown-linux-gnu < %s | FileCheck %s

	.text
add:
	add 3, 4, 3
	blr
	.gnu_attribute 4, 13

# CHECK-LABEL: add:
# CHECK: .gnu_attribute 4, 13
