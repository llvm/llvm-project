// RUN: %llvmgcc -xc %s -c -o - | llvm-dis | grep getelementptr

char *test(char* C) {
  return C-1;   // Should turn into a GEP
}

int *test2(int* I) {
  return I-1;
}
