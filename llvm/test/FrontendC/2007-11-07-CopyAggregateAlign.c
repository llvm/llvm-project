// RUN: %llvmgcc -S %s -o - | grep "align 2" | count 6
struct A { char s, t, u, v; short a; };
void q() { struct A a, b; a = b; }
