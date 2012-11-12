// RUN: llvm-mc -filetype=obj -triple i686-pc-mingw32 %s | coff-dump.py | FileCheck %s

	.def	 _foobar;
	.scl	2;
	.type	32;
	.endef
	.text
	.long   0
	.globl	_foobar
	.align	16, 0x90
_foobar:                                # @foobar
# BB#0:
	ret

	.data
	.globl	_rust_crate             # @rust_crate
	.align	4
_rust_crate:
	.long   0
	.long   _foobar
	.long	_foobar-_rust_crate
	.long	_foobar-_rust_crate

// CHECK:      Name                     = .data
// CHECK:      SectionData              =
// CHECK-NEXT:   00 00 00 00 00 00 00 00 - 1C 00 00 00 20 00 00 00 |............ ...|
// CHECK:        Relocations              = [
// CHECK-NEXT:   0 = {
// CHECK-NEXT:     VirtualAddress           = 0x4
// CHECK-NEXT:     SymbolTableIndex         =
// CHECK-NEXT:     Type                     = IMAGE_REL_I386_DIR32 (6)
// CHECK-NEXT:     SymbolName               = _foobar
// CHECK-NEXT:   }
// CHECK-NEXT:   1 = {
// CHECK-NEXT:     VirtualAddress           = 0x8
// CHECK-NEXT:     SymbolTableIndex         = 0
// CHECK-NEXT:     Type                     = IMAGE_REL_I386_REL32 (20)
// CHECK-NEXT:     SymbolName               = .text
// CHECK-NEXT:   }
// CHECK-NEXT:   2 = {
// CHECK-NEXT:     VirtualAddress           = 0xC
// CHECK-NEXT:     SymbolTableIndex         = 0
// CHECK-NEXT:     Type                     = IMAGE_REL_I386_REL32 (20)
// CHECK-NEXT:     SymbolName               = .text
// CHECK-NEXT:   }
// CHECK-NEXT: ]
