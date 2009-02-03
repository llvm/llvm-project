// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

struct foo {
  unsigned int I:1;
  unsigned char J[1][123];
  unsigned int K:1;
 };

struct foo F;
