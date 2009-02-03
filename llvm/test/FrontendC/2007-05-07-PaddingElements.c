// PR 1278
// RUN: %llvmgcc %s -S -emit-llvm -O0 -o - | not grep "4 x i8] zeroinitializer"
// RUN: %llvmgcc %s -S -emit-llvm -O0 -o - | not grep "i32 0, i32 2"
struct s {
  double d1;
  int s1;
};

struct s foo(void) {
  struct s S = {1.1, 2};
  return S;
}
