// RUN: llvm-mc -triple aarch64-windows -filetype obj -o %t.obj %s
// RUN: llvm-objdump -d -r %t.obj | FileCheck %s

.section sec00, "ax"
nop
nop
nop
.section sec01, "ax"
.balign 4
nop
nop
nop

// CHECK: 0000000000000000 <sec00>:
// CHECK-NEXT: 0: d503201f      nop
// CHECK-NEXT: 4: d503201f      nop
// CHECK-NEXT: 8: d503201f      nop
// CHECK: 0000000000000000 <sec01>:
// CHECK-NEXT: 0: d503201f      nop
// CHECK-NEXT: 4: d503201f      nop
// CHECK-NEXT: 8: d503201f      nop
