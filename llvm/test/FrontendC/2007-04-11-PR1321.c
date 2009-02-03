// RUN: %llvmgcc %s -S -o /dev/null

struct X {
  unsigned int e0 : 17;
  unsigned int e1 : 17;
  unsigned int e2 : 17;
  unsigned int e3 : 17;
  unsigned int e4 : 17;
  unsigned int e5 : 17;
  unsigned int e6 : 17;
  unsigned int e7 : 17;
} __attribute__((packed)) x;
