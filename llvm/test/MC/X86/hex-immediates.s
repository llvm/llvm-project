# RUN: llvm-mc -filetype=obj %s -triple=x86_64-apple-darwin9 | llvm-objdump -d --print-imm-hex - | FileCheck %s

# CHECK: movabsq	$0x7fffffffffffffff, %rcx
movabsq	$0x7fffffffffffffff, %rcx
# CHECK: leaq	0x3e2(%rip), %rdi
leaq	0x3e2(%rip), %rdi
# CHECK: subq	$0x40, %rsp
subq	$0x40, %rsp
# CHECK: leal	(,%r14,4), %eax
leal	(,%r14,4), %eax
