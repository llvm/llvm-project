# The assembly below does NOT include the usual .subsections_via_symbols
# directive. Check that when the directive is absent we only create one block
# to cover the whole data section.
#
# REQUIRES: asserts
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj -o %t %s
# RUN: llvm-jitlink -debug-only=jitlink -noexec %t 2>&1 | FileCheck %s

# CHECK:        Creating graph symbols...
# CHECK:          Graphifying regular section __DATA,__data...
# CHECK-NEXT:       Creating block {{.*}} with 2 symbol(s)...
# CHECK-NEXT:         0x0000000000000004 -- 0x0000000000000008: _b
# CHECK-NEXT:         0x0000000000000000 -- 0x0000000000000008: _main

	.section	__DATA,__data
	.p2align	2
        .globl  _main
_main:
	.long	1
_b:
        .long   2
