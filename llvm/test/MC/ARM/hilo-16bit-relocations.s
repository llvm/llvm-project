@ RUN: llvm-mc %s -triple armv7-apple-darwin | FileCheck %s
@ RUN: llvm-mc %s -triple armv7-apple-darwin | FileCheck %s        
        
_t:
        movw    r0, :lower16:(L_foo$non_lazy_ptr - (L1 + 8))
        movt    r0, :upper16:(L_foo$non_lazy_ptr - (L1 + 8))
L1:

@ CHECK: movw	r0, :lower16:(L_foo$non_lazy_ptr-(L1+8))
@ CHECK: movt	r0, :upper16:(L_foo$non_lazy_ptr-(L1+8))
        
        .comm	_foo,4,2

	.section	__DATA,__nl_symbol_ptr,non_lazy_symbol_pointers
	.align	2
L_foo$non_lazy_ptr:
	.indirect_symbol	_foo
	.long	0
        
.subsections_via_symbols
