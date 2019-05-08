// RUN: llvm-mc %s -filetype=obj -triple=x86_64-pc-linux -o %t.o

// Check we print the address of `foo` and `bar`.
// RUN: llvm-objdump -d %t.o | FileCheck %s
// CHECK:      0000000000000000 foo:
// CHECK-NEXT:   0: {{.*}}  nop
// CHECK-NEXT:   1: {{.*}}  nop
// CHECK:      0000000000000002 bar:
// CHECK-NEXT:   2: {{.*}}  nop

// Check we do not print the addresses with -no-leading-addr.
// RUN: llvm-objdump -d --no-leading-addr %t.o | FileCheck %s --check-prefix=NOADDR
// NOADDR:      {{^}}foo:
// NOADDR-NEXT:   {{.*}} nop
// NOADDR-NEXT:   {{.*}} nop
// NOADDR:      {{^}}bar:
// NOADDR-NEXT:   {{.*}} nop

.text
.globl  foo
.type   foo, @function
foo:
 nop
 nop

bar:
 nop
