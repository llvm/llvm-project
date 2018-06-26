# RUN: llvm-mc -triple x86_64-unknown-unknown -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s
	
.section .rodata, "ax"
# CHECK: warning: setting incorrect section attributes for .rodata
nop

.section .rodata, "a"
# CHECK-NOT: warning:
nop

