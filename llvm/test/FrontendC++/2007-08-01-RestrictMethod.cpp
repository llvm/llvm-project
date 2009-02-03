// RUN: %llvmgxx -c -emit-llvm %s -o - | llvm-dis | grep noalias


class foo {
  int member[4];
  
  void bar(int * a);
  
};

void foo::bar(int * a) __restrict {
  member[3] = *a;
}
