// RUN: llvm-mc -triple mips-unknown-unknown %s

i:
        .long    g
g = h
h = i
