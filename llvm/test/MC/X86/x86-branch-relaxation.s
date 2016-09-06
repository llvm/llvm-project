# RUN: llvm-mc -filetype=obj -triple=i386-unknown-unknown %s -o %t
# RUN: llvm-objdump -r -D -section .text.bar -triple i386-unknown-unknown-code16 %t | FileCheck --check-prefix=CHECK16 %s
# RUN: llvm-objdump -r -D -section .text.baz -triple i386-unknown-unknown        %t | FileCheck --check-prefix=CHECK32 %s 	
	.text
	.section	.text.foo,"",@progbits

	.code16
	.globl	foo
foo:
	nop

	.section	.text.bar,"",@progbits
	.globl	bar16
bar16:
	jmp foo

	.section	.text.baz,"",@progbits
	.code32
	.globl	baz32
baz32:
	jmp foo
	


	
// CHECK16-LABEL: bar16
// CHECK16-NEXT: e9 fe ff 	jmp	-2 <bar16+0x1>
// CHECK32-LABEL: baz32
// CHECK32-NEXT: e9 fc ff ff ff 	jmp	-4 <baz32+0x1>

	
