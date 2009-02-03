// RUN: %llvmgxx -O0 -emit-llvm -S -g -o - %s | grep {i32 1,}
// RUN: %llvmgxx -O0 -emit-llvm -S -g -o - %s | grep {i32 2,}
class A {
public:
  int x;
protected:
  int y;
private:
  int z;
};

A a;
