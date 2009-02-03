// RUN: %llvmgcc -S %s -o - 
// PR1662

int foo(unsigned char *test) {
  return 0U - (unsigned int )test;
}

