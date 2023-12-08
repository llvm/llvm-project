// RUN: llvm-mc -triple x86_64-apple-darwin10 %s 2> %t.err > %t
// RUN: FileCheck --check-prefix=CHECK-OUTPUT < %t %s
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t.err %s

.macro test_macro reg1, reg2
mov $1, %eax
mov $2, %eax
.endmacro
test_macro %ebx, %ecx

// CHECK-ERROR: 5:1: warning: macro defined with named parameters which are not used in macro body, possible positional parameter found in body which will have no effect

// CHECK-OUTPUT: movl	$1, %eax
// CHECK-OUTPUT: movl	$2, %eax
