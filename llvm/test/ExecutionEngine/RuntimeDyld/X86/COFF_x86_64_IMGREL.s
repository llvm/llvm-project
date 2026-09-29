# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-pc-win32 -filetype=obj -o %t/COFF_x86_64_IMGREL.o %s
# RUN: llvm-rtdyld -triple=x86_64-pc-win32 -verify -target-addr-start=40960000000000 -check=%s %t/COFF_x86_64_IMGREL.o
.text
	.def	 F;
	.scl	2;
	.type	32;
	.endef
	.globl	__constdata

.section    .rdata, "dr", discard, __constdata
    .align	8
    __constdata:
	    .quad	0

.text
	.globl	F
	.align	16, 0x90

F:                                      # @F
# rtdyld-check: decode_operand(inst1, 3) = section_addr(COFF_x86_64_IMGREL.o, .text)+0-40960000000000
inst1:
    mov %ebx, F@IMGREL
# rtdyld-check: decode_operand(inst2, 3) = section_addr(COFF_x86_64_IMGREL.o, .rdata)+5-40960000000000
inst2:
    mov %ebx, (__constdata@imgrel+5)
        .data
        .space 375
rel1:
# rtdyld-check: *{4}rel1 = string - section_addr(COFF_x86_64_IMGREL.o, .data)
	.secrel32 string

# We explicitly add padding to put string outside of the 16bit address space
# (absolute and as an offset from .data), so that relocations involving
# 32bit addresses / offsets are not accidentally truncated to 16 bits.
        .space 65536
        .global string
        .align 1
string:
        .asciz "Hello World\n"
