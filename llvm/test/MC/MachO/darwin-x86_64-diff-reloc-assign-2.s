// RUN: llvm-mc -triple x86_64-apple-darwin9 %s -filetype=obj -o - | llvm-readobj -r --expand-relocs - | FileCheck %s

	.data
L_var1:
L_var2:
	.long L_var2 - L_var1
	.set L_var3, .
	.set L_var4, .
	.long L_var4 - L_var3

// CHECK:      Relocations [
// CHECK-NEXT:   Section __data {
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x4
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_SUBTRACTOR (5)
// CHECK-NEXT:       Section: __data (2)
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x4
// CHECK-NEXT:       PCRel: 0
// CHECK-NEXT:       Length: 2
// CHECK-NEXT:       Type: X86_64_RELOC_UNSIGNED (0)
// CHECK-NEXT:       Section: __data (2)
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]
