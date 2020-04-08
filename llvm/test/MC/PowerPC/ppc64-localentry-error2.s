
# RUN: not llvm-mc -triple powerpc64-unknown-unknown -filetype=obj < %s 2> %t
# RUN: FileCheck < %t %s
# RUN: not llvm-mc -triple powerpc64le-unknown-unknown -filetype=obj < %s 2> %t
# RUN: FileCheck < %t %s

	.globl remote_sym
sym:
	.localentry sym, remote_sym

# CHECK: error: .localentry expression must be absolute.

