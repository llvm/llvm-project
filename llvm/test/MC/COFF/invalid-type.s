# RUN: not llvm-mc -triple i686-windows -filetype obj -o /dev/null %s 2>&1 | FileCheck %s

# CHECK: symbol type specified outside of a symbol definition
# CHECK: symbol type specified outside of a symbol definition
	.type 65536
	.type 65537

