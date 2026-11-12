
# RUN: not llvm-mc -triple powerpc64-unknown-unknown -filetype=obj < %s 2> %t
# RUN: FileCheck < %t %s
# RUN: not llvm-mc -triple powerpc64le-unknown-unknown -filetype=obj < %s 2> %t
# RUN: FileCheck < %t %s

	.globl remote_sym
sym:

# CHECK: :0: error: .localentry expression must be a power of 2
	.localentry sym, 123

# CHECK: :[[#@LINE+1]]:19: error: .localentry expression must be absolute
	.localentry sym, remote_sym

