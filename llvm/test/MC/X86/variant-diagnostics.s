# RUN: not llvm-mc -triple i386-linux-gnu -filetype asm -o /dev/null 2>&1 %s \
# RUN:   | FileCheck %s

	.text

function:
	call external@invalid

# CHECK: error: invalid variant 'invalid'
# CHECK: 	call external@invalid
# CHECK:                      ^
