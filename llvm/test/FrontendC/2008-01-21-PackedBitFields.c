// RUN: %llvmgcc %s -S -o -

typedef double Al1Double __attribute__((aligned(1)));
struct x { int a:23; Al1Double v; };
struct x X = { 5, 3.0 };
double foo() { return X.v; }

