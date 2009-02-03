// RUN: %llvmgxx -S %s -o -

#include <new>
typedef double Ty[4];

void foo(Ty *XX) {
  new(XX) Ty();
}
