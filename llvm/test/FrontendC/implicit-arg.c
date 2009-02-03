// RUN: %llvmgcc %s -S -emit-llvm -O0 -o -
// RUN: %llvmgcc %s -S -emit-llvm -O1 -o -
// rdar://6518089

static int bar();
void foo() {
  int a = bar();
}
int bar(unsigned a) {
}
