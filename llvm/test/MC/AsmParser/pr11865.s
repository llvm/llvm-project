// RUN: llvm-mc -triple i386-unknown-unknown %s

i:
        .long    g
g = h
h = i
