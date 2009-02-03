// RUN: %llvmgcc %s -c -o - | llvm-dis | grep llvm.global_ctors

void foo() __attribute__((constructor));
void foo() {
  bar();
}
