// RUN: llvm-mc -triple x86_64-apple-darwin9 %s -filetype=obj -o - | llvm-readobj -r --expand-relocs - | FileCheck %s

	.data
L_var1:
L_var2:
	.long L_var2 - L_var1
	.set L_var3, .
	.set L_var4, .
	.long L_var4 - L_var3

// CHECK:      Relocations [
// CHECK-NEXT: ]
