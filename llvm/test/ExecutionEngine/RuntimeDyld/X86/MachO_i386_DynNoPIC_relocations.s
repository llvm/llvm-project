# RUN: llvm-mc -triple=i386-apple-macosx10.4 -filetype=obj -o %T/test_i386.o %s
# RUN: llvm-rtdyld -triple=i386-apple-macosx10.4 -verify -check=%s %/T/test_i386.o

// Put the section used in the test at a non zero address.
	.long 4

	.section	__TEXT,__text2,regular,pure_instructions
	.globl	bar
	.align	4, 0x90
bar:
	calll	tmp0$pb
tmp0$pb:
	popl	%eax
# Test section difference relocation to non-lazy ptr section.
# rtdyld-check: decode_operand(inst1, 4) = x$non_lazy_ptr - tmp0$pb + 8
inst1:
	movl	(x$non_lazy_ptr-tmp0$pb)+8(%eax), %eax
        movl    (%eax), %ebx

# Test VANILLA relocation to jump table.
# rtdyld-check: decode_operand(inst2, 0) = bling$stub - next_pc(inst2)
inst2:
        calll	bling$stub
        addl    %ebx, %eax

# Test scattered VANILLA relocations.
inst3:
        movl    y+4, %ecx
        addl    %ecx, %eax
	retl

	.section	__IMPORT,__jump_table,symbol_stubs,pure_instructions+self_modifying_code,5
bling$stub:
	.indirect_symbol	bling
	.ascii	"\364\364\364\364\364"

	.section	__IMPORT,__pointers,non_lazy_symbol_pointers
x$non_lazy_ptr:
	.indirect_symbol	x
	.long	0

        .comm   x,4,2
        .comm   bling,4,2

        .globl	y
.zerofill __DATA,__common,y,8,3

.subsections_via_symbols
