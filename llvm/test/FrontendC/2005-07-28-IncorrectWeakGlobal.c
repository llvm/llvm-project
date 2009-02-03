// RUN: %llvmgcc %s -S -o - | grep TheGlobal | not grep weak

extern int TheGlobal;
int foo() { return TheGlobal; }
int TheGlobal = 1;
