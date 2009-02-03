// RUN: %llvmgxx %s -S -o - | not grep llvm.global_ctors
// This testcase corresponds to PR509
struct Data {
  unsigned *data;
  unsigned array[1];
};

Data shared_null = { shared_null.array };

