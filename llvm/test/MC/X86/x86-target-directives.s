# RUN: not llvm-mc -triple i386 -filetype asm -o - %s 2>&1 | FileCheck %s

	.code42

# CHECK: unknown directive .code42
# CHECK-NOT: unknown directive

