# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-freebsd13.0 %s -o %t
# RUN: llvm-objdump -t %t | FileCheck %s

# CHECK: 0000000000000000 gw    F .text  00000000 0x60 __impl_foo
# CHECK: 0000000000000000 g     F .text  00000000 0x60 foo
# CHECK: 0000000000000000 gw    F .text  00000000 0x60 foo@FBSD_1.1

.globl foo
.type foo,@function
foo:
  nop
  nop
  .localentry foo, 8

.symver __impl_foo, foo@FBSD_1.1
.weak   __impl_foo
.set    __impl_foo, foo
