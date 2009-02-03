// RUN: %llvmgcc %s -S -emit-llvm -O0 -o - | grep sret | count 5

struct abc {
 long a;
 long b;
 long c;
};
 
struct abc foo1(void);
struct abc foo2();

void bar() {
  struct abc dummy1 = foo1();
  struct abc dummy2 = foo2();
}
