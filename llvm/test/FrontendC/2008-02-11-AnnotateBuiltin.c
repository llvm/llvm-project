// RUN: %llvmgcc %s -S -o - | llvm-as | llvm-dis | grep llvm.annotation

int main() {
  int x = 0;
  return __builtin_annotation(x, "annotate");
}

