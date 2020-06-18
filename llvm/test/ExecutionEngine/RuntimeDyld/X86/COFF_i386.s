// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-mc -triple i686-windows -filetype obj -o %t/COFF_i386.o %s
// RUN: llvm-rtdyld -triple i686-windows -dummy-extern _printf=0x7ffffffd \
// RUN:   -dummy-extern _ExitProcess=0x7fffffff \
// RUN:   -verify -check=%s %t/COFF_i386.o

	.text

	.def _main
		.scl 2
		.type 32
	.endef
	.global _main
_main:
rel1:
	call _function				// IMAGE_REL_I386_REL32
# rtdyld-check: decode_operand(rel1, 0) = (_function-_main-4-1)
	xorl %eax, %eax
rel12:
	jmp _printf
# rtdyld-check: decode_operand(rel12, 0)[31:0] = (_printf-_main-4-8)

	.def _function
		.scl 2
		.type 32
	.endef
_function:
rel2:
	pushl string
# rtdyld-check: decode_operand(rel3, 3) = \
# rtdyld-check:   stub_addr(COFF_i386.o/.text, __imp__ExitProcess)
# rtdyld-check: *{4}(stub_addr(COFF_i386.o/.text, __imp__ExitProcess)) = \
# rtdyld-check:   _ExitProcess
rel3:
	calll *__imp__ExitProcess       	// IMAGE_REL_I386_DIR32

	.data

	.global relocations
relocations:
rel5:
	.long _function@imgrel			// IMAGE_REL_I386_DIR32NB
# rtdyld-check: *{4}rel5 = _function - section_addr(COFF_i386.o, .text)
rel6:
# rtdyld-check: *{2}rel6 = 1
	.secidx rel5                            // IMAGE_REL_I386_SECTION
rel7:
# rtdyld-check: *{4}rel7 = string - section_addr(COFF_i386.o, .data)
	.secrel32 string			// IMAGE_REL_I386_SECREL

# Test that addends work.
rel8:
# rtdyld-check: *{4}rel8 = string
	.long string				// IMAGE_REL_I386_DIR32
rel9:
# rtdyld-check: *{4}rel9 = string+1
	.long string+1				// IMAGE_REL_I386_DIR32
rel10:
# rtdyld-check: *{4}rel10 = string - section_addr(COFF_i386.o, .text) + 1
	.long string@imgrel+1			// IMAGE_REL_I386_DIR32NB
rel11:
# rtdyld-check: *{4}rel11 = string - section_addr(COFF_i386.o, .data) + 1
	.long string@SECREL32+1			// IMAGE_REL_I386_SECREL

# We explicitly add padding to put string outside of the 16bit address space
# (absolute and as an offset from .data), so that relocations involving
# 32bit addresses / offsets are not accidentally truncated to 16 bits.
	.space 65536
	.global string
	.align 1
string:
	.asciz "Hello World!\n"
