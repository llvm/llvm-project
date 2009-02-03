// PR1602
// RUN: %llvmgxx -c -emit-llvm %s -o - -O3 | llvm-dis | not grep ptrtoint
// RUN: %llvmgxx -c -emit-llvm %s -o - -O3 | llvm-dis | grep getelementptr | count 1


struct S { virtual void f(); };

typedef void (S::*P)(void);

const P p = &S::f; 

void g(S s) {
   (s.*p)();
 }
