// RUN: not llvm-mc -filetype=asm -triple x86_64-windows %s -o %t 2>%t.out
// RUN: FileCheck -input-file=%t.out %s

foo:
.seh_proc foo
// CHECK: Unfinished frame
