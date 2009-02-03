// RUN: %llvmgxx -c -emit-llvm %s -o - | llvm-dis | grep noalias

void foo(int * __restrict myptr1, int * myptr2) {
  myptr1[0] = 0;
  myptr2[0] = 0;
}
