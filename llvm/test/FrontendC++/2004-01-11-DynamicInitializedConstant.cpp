// RUN: %llvmgcc -xc++ -S -o - %s | not grep { constant }

extern int X;
const int Y = X;
const int* foo() { return &Y; }

