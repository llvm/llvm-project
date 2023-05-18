# RUN: llvm-mc -triple i386-pc-linux-gnu %s | FileCheck %s

	.bss
# CHECK: .bss

	.data.rel.ro
# CHECK: .data.rel.ro

	.data.rel
# CHECK: .data.rel

	.eh_frame
# CHECK: .eh_frame

	.rodata
# CHECK: .rodata

	.tbss
# CHECK: .tbss

	.tdata
# CHECK: .tdata

