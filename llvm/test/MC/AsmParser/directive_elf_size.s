# RUN: llvm-mc -triple i386-pc-linux-gnu %s | FileCheck %s

a:
	ret
.Lt:
# CHECK: .size	a, .Lt-a
	.size	a, .Lt-a

