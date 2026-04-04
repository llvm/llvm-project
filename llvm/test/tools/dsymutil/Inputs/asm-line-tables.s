.text
.globl _foo
.p2align 2
_foo:
    ret
.globl start
.p2align 2
start:
    bl _foo
    ret
