// RUN: %llvmgxx -S %s -o - | llvm-as -f -o /dev/null

struct C {};

C &foo();

void foox() {
  for (; ; foo());
}

