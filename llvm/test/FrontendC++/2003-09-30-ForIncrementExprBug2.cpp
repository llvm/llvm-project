// RUN: %llvmgxx -S %s -o - | llvm-as -f -o /dev/null

// Test with an opaque type

struct C;

C &foo();

void foox() {
  for (; ; foo());
}

