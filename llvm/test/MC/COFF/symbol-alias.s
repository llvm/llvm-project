// The purpose of this test is to verify that symbol aliases
// (@foo = alias <type> @bar) generate the correct entries in the symbol table.
// They should be identical except for the name.

// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s | coff-dump.py | FileCheck %s
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s | coff-dump.py | FileCheck %s

	.def	 _foo;
	.scl	2;
	.type	32;
	.endef
	.text
	.globl	_foo
	.align	16, 0x90
_foo:                                   # @foo
# BB#0:                                 # %entry
	ret

	.data
	.globl	_bar                    # @bar
	.align	4
_bar:
	.long	0                       # 0x0


	.globl	_foo_alias
_foo_alias = _foo
	.globl	_bar_alias
_bar_alias = _bar

// CHECK:      Name               = {{_?}}foo
// CHECK-NEXT: Value              = [[FOO_VALUE:.*$]]
// CHECK-NEXT: SectionNumber      = [[FOO_SECTION_NUMBER:.*$]]
// CHECK-NEXT: SimpleType         = [[FOO_SIMPLE_TYPE:.*$]]
// CHECK-NEXT: ComplexType        = [[FOO_COMPLEX_TYPE:.*$]]
// CHECK-NEXT: StorageClass       = [[FOO_STORAGE_CLASS:.*$]]
// CHECK-NEXT: NumberOfAuxSymbols = [[FOO_NUMBER_OF_AUX_SYMBOLS:.*$]]

// CHECK:      Name               = {{_?}}bar
// CHECK-NEXT: Value              = [[BAR_VALUE:.*$]]
// CHECK-NEXT: SectionNumber      = [[BAR_SECTION_NUMBER:.*$]]
// CHECK-NEXT: SimpleType         = [[BAR_SIMPLE_TYPE:.*$]]
// CHECK-NEXT: ComplexType        = [[BAR_COMPLEX_TYPE:.*$]]
// CHECK-NEXT: StorageClass       = [[BAR_STORAGE_CLASS:.*$]]
// CHECK-NEXT: NumberOfAuxSymbols = [[BAR_NUMBER_OF_AUX_SYMBOLS:.*$]]

// CHECK:      Name               = {{_?}}foo_alias
// CHECK-NEXT: Value              = [[FOO_VALUE]]
// CHECK-NEXT: SectionNumber      = [[FOO_SECTION_NUMBER]]
// CHECK-NEXT: SimpleType         = [[FOO_SIMPLE_TYPE]]
// CHECK-NEXT: ComplexType        = [[FOO_COMPLEX_TYPE]]
// CHECK-NEXT: StorageClass       = [[FOO_STORAGE_CLASS]]
// CHECK-NEXT: NumberOfAuxSymbols = [[FOO_NUMBER_OF_AUX_SYMBOLS]]

// CHECK:      Name               = {{_?}}bar_alias
// CHECK-NEXT: Value              = [[BAR_VALUE]]
// CHECK-NEXT: SectionNumber      = [[BAR_SECTION_NUMBER]]
// CHECK-NEXT: SimpleType         = [[BAR_SIMPLE_TYPE]]
// CHECK-NEXT: ComplexType        = [[BAR_COMPLEX_TYPE]]
// CHECK-NEXT: StorageClass       = [[BAR_STORAGE_CLASS]]
// CHECK-NEXT: NumberOfAuxSymbols = [[BAR_NUMBER_OF_AUX_SYMBOLS]]

