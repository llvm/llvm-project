// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s | coff-dump.py | FileCheck %s

// check that we produce the correct relocation for .secrel32

Lfoo:
	.secrel32	Lfoo

// CHECK:       Relocations              = [
// CHECK-NEXT:    0 = {
// CHECK-NEXT:       VirtualAddress           = 0x0
// CHECK-NEXT:       SymbolTableIndex         = 0
// CHECK-NEXT:       Type                     = IMAGE_REL_I386_SECREL (11)
// CHECK-NEXT:       SymbolName               = .text
// CHECK-NEXT:     }
