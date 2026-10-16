# RUN: rm -fr %t && split-file %s %t && cd %t
# RUN: not llvm-mc -triple=x86_64 a.s 2>&1 | FileCheck a.s
# RUN: not llvm-mc -triple=x86_64 -filetype=obj b.s 2>&1 | FileCheck b.s
# RUN: not llvm-mc -triple=x86_64 -filetype=obj c.s 2>&1 | FileCheck c.s

#--- a.s
.section .text.f1,"ax",@progbits
# CHECK: [[#@LINE+1]]:12: error: log2 alignment must be in the range [0, 63]
.prefalign 64

# CHECK: [[#@LINE+1]]:13: error: expected comma
.prefalign 4

# CHECK: [[#@LINE+1]]:14: error: expected symbol name
.prefalign 4,

# CHECK: [[#@LINE+1]]:22: error: expected comma
.prefalign 4,.text.f1

# CHECK: [[#@LINE+1]]:23: error: expected absolute expression
.prefalign 4,.text.f1,trap

# CHECK: [[#@LINE+1]]:23: error: fill value must be in range [0, 255]
.prefalign 4,.text.f1,256

# CHECK: [[#@LINE+1]]:23: error: fill value must be in range [0, 255]
.prefalign 4,.text.f1,-1

## Non-zero fill in a BSS section.
.bss
# CHECK: [[#@LINE+1]]:19: error: non-zero fill in BSS section '.bss'
.prefalign 4,.Lend,1
# CHECK: [[#@LINE+1]]:19: error: non-zero fill in BSS section '.bss'
.prefalign 4,.Lend,nop
.space 1
.Lend:

#--- b.s
## End symbol is undefined.
.section .text.f1,"ax",@progbits
# CHECK: <unknown>:0: error: .prefalign end symbol 'undef' must be in the current section
.prefalign 4,undef,0

#--- c.s
## End symbol is defined in a different section.
.section .text.f1,"ax",@progbits
.prefalign 4,.Lend,0
# CHECK: <unknown>:0: error: .prefalign end symbol '.Lend' must be in the current section
.section .text.f2,"ax",@progbits
.Lend:
