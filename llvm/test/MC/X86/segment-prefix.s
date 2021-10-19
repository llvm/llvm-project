# RUN: llvm-mc %s -triple x86_64-linux-gnu -filetype=obj -o - | llvm-objdump -d - | FileCheck %s

.text
.global foo
foo:
	cs outsl
	ds outsl
	es outsw
	fs outsw
	gs outsl
	ss outsl
	retq

# CHECK: <foo>:
# CHECK-NEXT: 2e 6f                         outsl  %cs:(%rsi), %dx
# CHECK-NEXT: 3e 6f                         outsl  %ds:(%rsi), %dx
# CHECK-NEXT: 26 66 6f                      outsw  %es:(%rsi), %dx
# CHECK-NEXT: 64 66 6f                      outsw  %fs:(%rsi), %dx
# CHECK-NEXT: 65 6f                         outsl  %gs:(%rsi), %dx
# CHECK-NEXT: 36 6f                         outsl  %ss:(%rsi), %dx
