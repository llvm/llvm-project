// The code generated for this testcase should be completely typesafe!
// RUN: %llvmgcc -xc++ -S -o - %s | llvm-as | opt -die | llvm-dis | \
// RUN:    notcast

struct contained {
  unsigned X;
  contained();
};

struct base {
  unsigned A, B;
};

struct derived : public base {
  contained _M_value_field;
};

int test() {
  derived X;
}

