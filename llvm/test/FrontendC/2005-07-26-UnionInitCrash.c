// PR607
// RUN: %llvmgcc %s -S -o -
union { char bytes[8]; double alignment; }EQ1 = {0,0,0,0,0,0,0,0};
