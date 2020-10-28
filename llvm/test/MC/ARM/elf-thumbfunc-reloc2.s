// RUN: llvm-mc %s -triple=thumbv7-linux-gnueabi \
// RUN: -filetype=obj -o - | llvm-readobj -S --sd -r --symbols - | \
// RUN: FileCheck %s

// We want to test relocatable thumb function call.

	.thumb_func
foo:
	.fnstart
	bx	lr
	.cantunwind
	.fnend

	.align	1
bar:
	.fnstart
	push	{r7, lr}
	bl	foo(PLT)
	pop	{r7, pc}
	.cantunwind
	.fnend

// make sure that bl 0 <foo> (fff7feff) is correctly encoded
// CHECK: Sections [
// CHECK:   SectionData (
// CHECK:     0000: 704780B5 FFF7FEFF 80BD
// CHECK:   )
// CHECK: ]

// CHECK:      Relocations [
// CHECK-NEXT:   Section {{.*}} .rel.text {
// CHECK-NEXT:     0x4 R_ARM_THM_CALL foo 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {{.*}} .rel.ARM.exidx {
// CHECK-NEXT:     0x0 R_ARM_PREL31 .text 0x0
// CHECK-NEXT:     0x8 R_ARM_PREL31 .text 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// make sure foo is thumb function: bit 0 = 1
// CHECK:      Symbols [
// CHECK:        Symbol {
// CHECK:          Name: foo
// CHECK-NEXT:     Value: 0x1
