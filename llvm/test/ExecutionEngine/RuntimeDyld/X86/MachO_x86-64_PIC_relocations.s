# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -relocation-model=pic -filetype=obj -o %T/test_x86-64.o %s
# RUN: llvm-rtdyld -triple=x86_64-apple-macosx10.9 -dummy-extern ds1=0xfffffffffffffffe -dummy-extern ds2=0xffffffffffffffff -verify -check=%s %/T/test_x86-64.o

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

# Test dummy-extern relocation.
# rtdyld-check: *{8}z1 = ds1
z1:
        .quad   ds1

# Test external-symbol relocation bypass: symbols with addr 0xffffffffffffffff
# don't have their relocations applied.
# rtdyld-check: *{8}z2 = 0
z2:
        .quad   ds2

# Test absolute symbols.
# rtdyld-check: abssym = 0xdeadbeef
        .globl  abssym
abssym = 0xdeadbeef

	# Test subtractor relocations.
# rtdyld-check: *{8}z3 = z4 - z5 + 4
z3:
        .quad  z4 - z5 + 4

        .section        __DATA,_tmp1
z4:
        .byte 1

        .section        __DATA,_tmp2
z5:
        .byte 1

.subsections_via_symbols
