// Test returning a single element aggregate value containing a double.
// RUN: %llvmgcc %s -S -o -

struct X {
  double D;
};

struct Y { 
  struct X x; 
};

struct Y bar();

void foo(struct Y *P) {
  *P = bar();
}

struct Y bar() {
  struct Y a;
  a.x.D = 0;
  return a;
}

