# RUN: llvm-mc %s -triple x86_64-linux-gnu -filetype=obj -o - | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc %s -triple x86_64-windows-msvc -filetype=obj -o - | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc %s -triple x86_64-apple-macos -filetype=obj -o - | llvm-objdump -d - | FileCheck %s

.text
.global foo
foo:
	.code64
	movl (%eax), %eax
	.code32
	movl (%eax), %eax
	.code16
	movl (%eax), %eax
	.code64
	retq

# CHECK: <foo>:
# CHECK-NEXT: 67 8b 00                      movl    (%eax), %eax
# CHECK-NEXT: 8b 00                         movl    (%rax), %eax
# CHECK-NEXT: 67 66 8b 00                   movw    (%eax), %ax
# CHECK-NEXT: c3                            retq
