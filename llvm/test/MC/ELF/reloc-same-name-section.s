// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux %s -o - | llvm-readobj -r --expand-relocs - | FileCheck %s

// test that we produce one relocation against each section.

// CHECK:      Relocations [
// CHECK-NEXT:   Section {{.*}} {
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset:
// CHECK-NEXT:       Type:
// CHECK-NEXT:       Symbol:  .foo (4)
// CHECK-NEXT:       Addend:
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset:
// CHECK-NEXT:       Type:
// CHECK-NEXT:       Symbol:  .foo (5)
// CHECK-NEXT:       Addend:
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]

	.section	.foo,"aG",@progbits,v,comdat
f:

	.section	.foo,"a",@progbits
g:


	.section	.bar
	.quad	f
	.quad	g
