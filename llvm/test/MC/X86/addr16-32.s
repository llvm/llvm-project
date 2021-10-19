# RUN: llvm-mc %s -triple x86_64-linux-gnu -filetype=obj -o - | llvm-objdump -d - | FileCheck %s

.text
.global foo
foo:
	insl
	gs outsl
	.code64
	addr32 insl
	addr32 gs outsl
	.code32
	addr16 insl
	addr16 gs outsl
	.code64
	retq

# CHECK: <foo>:
# CHECK-NEXT: 6d                            insl   %dx, %es:(%rdi)
# CHECK-NEXT: 65 6f                         outsl  %gs:(%rsi), %dx
# CHECK-NEXT: 67 6d                         insl   %dx, %es:(%edi)
# CHECK-NEXT: 67 65 6f                      outsl  %gs:(%esi), %dx
# CHECK-NEXT: 67 6d                         insl   %dx, %es:(%edi)
# CHECK-NEXT: 67 65 6f                      outsl  %gs:(%esi), %dx
