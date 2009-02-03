// RUN: %llvmgxx -xc++ %s -c -o - | llvm-dis | grep getelementptr

struct foo {
  int array[100];
  void *getAddr(unsigned i);
};

void *foo::getAddr(unsigned i) {
  return &array[i];
}
