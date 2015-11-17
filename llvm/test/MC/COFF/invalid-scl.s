# RUN: not llvm-mc -triple i686-windows -filetype obj -o /dev/null %s 2>&1 | FileCheck %s

# CHECK: storage class specified outside of symbol definition
# CHECK: storage class specified outside of symbol definition
	.scl 1337
	.scl 1337

