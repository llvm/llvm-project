# RUN: not llvm-mc -triple i686-windows -filetype obj -o /dev/null %s 2>&1 | FileCheck %s

# CHECK: error: starting a new symbol definition without completing the previous one
# CHECK: error: starting a new symbol definition without completing the previous one
	.def first
	.def second
	.def third

