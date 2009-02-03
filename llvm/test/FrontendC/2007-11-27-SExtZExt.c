// RUN: %llvmgcc -S %s -emit-llvm -o - | grep "signext" | count 4

signed char foo1() { return 1; }

void foo2(signed short a) { }

signed char foo3(void) { return 1; }

void foo4(a) signed short a; { }



