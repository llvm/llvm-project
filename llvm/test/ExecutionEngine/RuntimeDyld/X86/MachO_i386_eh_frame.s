# RUN: llvm-mc -triple=i386-apple-macosx10.4 -filetype=obj -o %T/MachO_i386_eh_frame.o %s
# RUN: llvm-rtdyld -triple=i386-apple-macosx10.4 -verify -map-section MachO_i386_eh_frame.o,__text=0x2000 -check=%s %/T/MachO_i386_eh_frame.o

# rtdyld-check: *{4}(section_addr(MachO_i386_eh_frame.o, __eh_frame) + 0x20) = (main - (section_addr(MachO_i386_eh_frame.o, __eh_frame) + 0x20))[31:0]
# rtdyld-check: *{4}(section_addr(MachO_i386_eh_frame.o, __eh_frame) + 0x24) = 0x9

	.section	__TEXT,__text,regular,pure_instructions

	.globl	bar
	.align	4, 0x90
bar:
        retl

        .globl	main
	.align	4, 0x90
main:
	.cfi_startproc
	pushl	%ebp
Ltmp0:
	.cfi_def_cfa_offset 8
Ltmp1:
	.cfi_offset %ebp, -8
	movl	%esp, %ebp
Ltmp2:
	.cfi_def_cfa_register %ebp
	popl	%ebp
	jmp	bar
	.cfi_endproc

.subsections_via_symbols
