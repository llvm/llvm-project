# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj -o %t/test_x86-64.o %s
# RUN: llvm-rtdyld -triple=x86_64-apple-macosx10.9 -dummy-extern ds1=0xfffffffffffffffe -dummy-extern ds2=0xffffffffffffffff -verify -check=%s %t/test_x86-64.o

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
# Verify the alignment of the GOT entry, the contents of the GOT entry for y,
# and that the movq instruction references the correct GOT entry address:
# rtdyld-check: stub_addr(test_x86-64.o/__text, y)[2:0] = 0
# rtdyld-check: *{8}(stub_addr(test_x86-64.o/__text, y)) = y
# rtdyld-check: decode_operand(insn3, 4) = stub_addr(test_x86-64.o/__text, y) - next_pc(insn3)
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

# Test subtractor relocations between named symbols.
# rtdyld-check: *{8}z3a = z4 - z5 + 4
z3a:
        .quad  z4 - z5 + 4

# Test subtractor relocations between anonymous symbols.
# rtdyld-check: *{8}z3b = (section_addr(test_x86-64.o, _tmp3) + 4) - (section_addr(test_x86-64.o, _tmp4)) + 8
z3b:
        .quad  Lanondiff_1 - Lanondiff_2 + 8

# Test subtractor relocations between named and anonymous symbols.
# rtdyld-check: *{8}z3c = z4 - (section_addr(test_x86-64.o, _tmp4)) + 12
z3c:
        .quad  z4 - Lanondiff_2 + 12

# Test subtractor relocations between anonymous and named symbols.
# rtdyld-check: *{8}z3d = (section_addr(test_x86-64.o, _tmp3) + 4) - z4 + 16
z3d:
        .quad  Lanondiff_1 - z4 + 16

        .section        __DATA,_tmp1
z4:
        .byte 1

        .section        __DATA,_tmp2
z5:
        .byte 1

        .section        __DATA,_tmp3
        .long 1         # padding to make sure we handle non-zero offsets.
Lanondiff_1:
        .byte 1

        .section        __DATA,_tmp4
Lanondiff_2:
        .byte 1

.subsections_via_symbols
