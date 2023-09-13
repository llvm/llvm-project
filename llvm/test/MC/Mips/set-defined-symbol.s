# RUN: llvm-mc %s -triple=mips -mcpu=mips32 -filetype=obj -o - | \
# RUN:   llvm-objdump -d -r - | FileCheck %s

  .global foo
  .weak bar
  .set bar, b
  .set foo, b
  .set foo, a
a:
  nop
# CHECK-NOT: a:
# CHECK: <foo>:

b:
  nop
# CHECK-NOT: b:
# CHECK-NOT: foo:
# CHECK: <bar>:
