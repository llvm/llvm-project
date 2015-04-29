# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -relocation-model=pic -filetype=obj -o %T/test_x86-64.o %s
# RUN: llvm-rtdyld -triple=x86_64-apple-macosx10.9 -verify -check=%s %/T/test_x86-64.o

        .section	__TEXT,__text,regular,pure_instructions
	.globl	foo
	.align	4, 0x90
foo:
        retq

	.globl	main
	.align	4, 0x90
main:
# Test PC-rel branch.
# rtdyld-check: decode_operand(insn1, 0) = foo - next_pc(insn1)
insn1:
        callq	foo

# Test PC-rel signed.
# rtdyld-check: decode_operand(insn2, 4) = x - next_pc(insn2)
insn2:
	movl	x(%rip), %eax

# Test PC-rel GOT relocation.
# Verify both the contents of the GOT entry for y, and that the movq instruction
# references the correct GOT entry address:
# rtdyld-check: *{8}(stub_addr(test_x86-64.o, __text, y)) = y
# rtdyld-check: decode_operand(insn3, 4) = stub_addr(test_x86-64.o, __text, y) - next_pc(insn3)
insn3:
        movq	y@GOTPCREL(%rip), %rax

        movl	$0, %eax
	retq

# Test processing of the __eh_frame section.
# rtdyld-check: *{8}(section_addr(test_x86-64.o, __eh_frame) + 0x20) = eh_frame_test - (section_addr(test_x86-64.o, __eh_frame) + 0x20)
eh_frame_test:
        .cfi_startproc
        retq
        .cfi_endproc

        .comm   y,4,2

        .section	__DATA,__data
	.globl	x
	.align	2
x:
        .long   5

.subsections_via_symbols
