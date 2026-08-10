// RUN: not llvm-mc -filetype=obj -triple i386-pc-linux-gnu %s -o /dev/null 2>&1 | FileCheck  %s

// CHECK: error: invalid variant 'BADRELOC'

        .text
foo:
	leal	.Lfoo@BADRELOC(%ebx), %eax
