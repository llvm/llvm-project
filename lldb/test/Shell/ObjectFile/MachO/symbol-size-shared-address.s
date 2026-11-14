# REQUIRES: x86

# With relocatable DWARF, every __DWARF section of an object file starts at
# address 0, like __text. A symbol without a size is still bounded by the end
# of the section it is defined in.

# RUN: llvm-mc -triple x86_64-apple-macosx -filetype=obj \
# RUN:   -macho-relocatable-dwarf %s -o %t.o
# RUN: lldb-test symtab --find-symbols-by-regex='.*' %t.o | FileCheck %s

# CHECK-DAG: Code 0x0000000000000000 0x0000000000000003 {{.*}} foo
# CHECK-DAG: Code 0x0000000000000003 0x0000000000000002 {{.*}} bar

	.text
	.globl	_foo
_foo:
	nop
	nop
	ret
	.globl	_bar
_bar:
	nop
	ret

	.section	__DWARF,__debug_str,regular,debug
	.space	0x100
