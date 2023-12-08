// RUN: llvm-mc -triple arm-unknown-unknown %s

i:
        .long    g
g = h
h = i
