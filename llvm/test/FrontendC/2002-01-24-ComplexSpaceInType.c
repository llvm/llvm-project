// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

// This caused generation of the following type name:
//   %Array = uninitialized global [10 x %complex int]
//
// which caused problems because of the space int the complex int type
//

struct { int X, Y; } Array[10];

void foo() {}
