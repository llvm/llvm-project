# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-pc-win32 -filetype=obj -o %t/COFF_x86_64_SECREL.o %s
# RUN: llvm-rtdyld -triple=x86_64-pc-win32 -verify -check=%s %t/COFF_x86_64_SECREL.o

	.section	.rdata,"dr"
sec_base:                               # section-relative offset 0
	.zero	0x10
target:                                 # section-relative offset 0x10
	.long	0

	.data
	.globl	relocations
relocations:

sr_symoff:
	.secrel32	target
# rtdyld-check: *{4}sr_symoff = 0x10

sr_addend:
	.secrel32	sec_base+0x2a
# rtdyld-check: *{4}sr_addend = 0x2a

sr_both:
	.secrel32	target+0x2a
# rtdyld-check: *{4}sr_both = 0x3a
