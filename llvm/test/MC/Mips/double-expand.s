# RUN: llvm-mc -triple=mipsel-unknown-linux < %s | FileCheck %s
# RUN: llvm-mc -triple=mipsel-unknown-linux < %s | \
# RUN:     llvm-mc -triple=mipsel-unknown-linux | FileCheck %s

# CHECK: bnez $2, foo
# CHECK: nop
# CHECK-NOT: nop

        .text
	bnez $2, foo
