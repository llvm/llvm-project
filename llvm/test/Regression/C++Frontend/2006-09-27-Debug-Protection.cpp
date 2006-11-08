// XFAIL: llvmgcc3
// RUN: %llvmgxx -O0 -emit-llvm -S -g -o - %s | grep 'uint 1,' &&
// RUN: %llvmgxx -O0 -emit-llvm -S -g -o - %s | grep 'uint 2,'

class A {
public:
  int x;
protected:
  int y;
private:
  int z;
};

A a;
