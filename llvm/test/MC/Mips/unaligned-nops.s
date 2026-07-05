# RUN: llvm-mc -filetype=obj  -triple=mipsel %s -o %t
.byte 1
.p2align 2
foo:
