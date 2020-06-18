
# RUN: not llvm-mc -triple powerpc64-unknown-unknown -filetype=obj < %s 2> %t
# RUN: FileCheck < %t %s
# RUN: not llvm-mc -triple powerpc64le-unknown-unknown -filetype=obj < %s 2> %t
# RUN: FileCheck < %t %s

sym:
	.localentry sym, 123

# CHECK: error: .localentry expression is not a valid power of 2.

