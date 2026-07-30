# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj -o %t/test_ELF_x86-64_PC8.o %s
# RUN: llvm-rtdyld -triple=x86_64-pc-linux -verify -map-section test_ELF_x86-64_PC8.o,.text.bar=0x10000 -map-section test_ELF_x86-64_PC8.o,.text.baz=0x10040 %t/test_ELF_x86-64_PC8.o
# RUN: llvm-rtdyld -triple=x86_64-pc-linux -verify -map-section test_ELF_x86-64_PC8.o,.text.baz=0x10000 -map-section test_ELF_x86-64_PC8.o,.text.bar=0x10040 %t/test_ELF_x86-64_PC8.o

# Test that R_X86_64_PC8 relocation works.

  .section .text.bar,"ax"
	.align	16, 0x90
	.type	bar,@function
bar:
	retq
.Ltmp1:
	.size	bar, .Ltmp1-bar

  .section .text.baz,"ax"
	.align	16, 0x90
	.type	baz,@function
baz:
  movq  %rdi, %rcx
  jrcxz bar
	retq
.Ltmp2:
	.size	baz, .Ltmp2-baz


	.section	".note.GNU-stack","",@progbits
