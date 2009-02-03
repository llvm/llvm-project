// RUN: %llvmgcc -O2 -c %s -o /dev/null
void f(_Complex float z);
void g() { f(1.0i); }
