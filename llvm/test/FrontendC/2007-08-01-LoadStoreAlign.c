// RUN: %llvmgcc -O3 -S -o - -emit-llvm %s | grep {align 1} | count 2
// RUN: %llvmgcc -O3 -S -o - -emit-llvm %s | llvm-as | llc

struct p {
  char a;
  int b;
} __attribute__ ((packed));

struct p t = { 1, 10 };
struct p u;

int main () {
  int tmp = t.b;
  u.b = tmp;
  return tmp;

}
