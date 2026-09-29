// RUN: llvm-mc -filetype=obj -triple i686-pc-mingw32 %s | llvm-readobj -S --sr --sd - | FileCheck %s

// COFF resolves differences between labels in the same section, unless that
// label is declared with function type.

.section baz, "xr"
	.globl	X
X:
	mov	Y-X+42,	%eax
	retl

	.globl	Y
Y:
	retl

	.def	 _foobar;
	.scl	2;
	.type	32;
	.endef
	.text
	.long   0
	.globl	_foobar
	.align	16, 0x90
_foobar:                                # @foobar
# %bb.0:
	ret

	.globl	_baz
_baz:
	calll	_foobar
	retl

	.data
	.globl	_rust_crate             # @rust_crate
	.align	4
_rust_crate:
	.long   0
	.long   _foobar
	.long	_foobar-_rust_crate
	.long	_foobar-_rust_crate

// Even though _baz and _foobar are in the same .text section, we keep the
// relocation for compatibility with the VC linker's /guard:cf and /incremental
// flags, even on mingw.

// CHECK:        Name: .text
// CHECK:        Relocations [
// CHECK-NEXT:     0x12 IMAGE_REL_I386_REL32 _foobar
// CHECK-NEXT:   ]

// CHECK:        Name: .data
// CHECK:        Relocations [
// CHECK-NEXT:     0x4 IMAGE_REL_I386_DIR32 _foobar
// CHECK-NEXT:     0x8 IMAGE_REL_I386_REL32 _foobar
// CHECK-NEXT:     0xC IMAGE_REL_I386_REL32 _foobar
// CHECK-NEXT:   ]
// CHECK:        SectionData (
// CHECK-NEXT:     0000: 00000000 00000000 0C000000 10000000
// CHECK-NEXT:   )

// CHECK:        Name: baz
// CHECK:        Relocations [
// CHECK-NEXT:   ]
// CHECK:        SectionData (
// CHECK-NEXT:     0000: A1300000 00C3C3
// CHECK-NEXT:   )
