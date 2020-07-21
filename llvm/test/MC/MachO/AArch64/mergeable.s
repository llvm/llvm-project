// RUN: llvm-mc -triple aarch64-apple-darwin14 %s -filetype=obj -o - | llvm-readobj -r --expand-relocs - | FileCheck %s
// FIXME: the final relocation should be internal, but the linker doesn't
// currently handle the it correctly.

// Test that we "S + K" produce a relocation with a symbol, but just S produces
// a relocation with the section.

	.section	__TEXT,__literal4,4byte_literals
L0:
	.long	42

	.section	__TEXT,__cstring,cstring_literals
L1:
	.asciz	"42"

	.section	__DATA,__data
	.quad	L0
	.quad	L0 + 1
	.quad	L1
	.quad	L1 + 1

// CHECK:      Relocations [
// CHECK-NEXT:   Section __data {
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x18
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: ARM64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Symbol: L1
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x10
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: ARM64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Symbol: L1
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x8
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: ARM64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Symbol: L0
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x0
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 3
// CHECK-NEXT:       Type: ARM64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Symbol: L0
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]
