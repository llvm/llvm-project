// RUN: not llvm-mc -triple x86_64-unknown-unknown %s > %t 2> %t.err
// RUN: FileCheck < %t %s
// RUN: FileCheck -check-prefix=CHECK-STDERR < %t.err %s

// CHECK: .long 1
.long 1
// CHECK-STDERR: invalid modifier 'GOTPCREL' (no symbols present)
.long 10 + 4@GOTPCREL
// CHECK: .long a@GOTPCREL+4
.long a + 4@GOTPCREL
// CHECK: .long a@GOTPCREL+b@GOTPCREL
.long (a + b)@GOTPCREL
// CHECK: .long (10+b@GOTPCREL)+4
.long 10 + b + 4@GOTPCREL
