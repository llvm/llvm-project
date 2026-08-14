# Test that pseudo-probe output does not depend on probe insertion order.
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -triple=x86_64 -filetype=obj %t/forward.s -o %t/forward.o
# RUN: llvm-mc -triple=x86_64 -filetype=obj %t/reverse.s -o %t/reverse.o
# RUN: cmp %t/forward.o %t/reverse.o

#--- forward.s
.text
.globl z_func
.type z_func,@function
z_func:
  nop
.globl a_func
.type a_func,@function
a_func:
  nop
.pseudoprobe 1 1 0 0 z_func
.pseudoprobe 2 1 0 0 a_func

#--- reverse.s
.text
.globl z_func
.type z_func,@function
z_func:
  nop
.globl a_func
.type a_func,@function
a_func:
  nop
.pseudoprobe 2 1 0 0 a_func
.pseudoprobe 1 1 0 0 z_func
