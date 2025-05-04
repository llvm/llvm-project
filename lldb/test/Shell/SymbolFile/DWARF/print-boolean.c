// RUN: %clang_host -O3 -ggdb -o %t %s
// RUN: %lldb %t \
// RUN:   -o "b 17" \
// RUN:   -o r \
// RUN:   -o "p t" \
// RUN:   -o exit | FileCheck %s

// CHECK: (lldb) p t
// CHECK-NEXT: (int) 1

int a, b, c;
int d(int e) { return e; }
int main() {
  int t;
  c = d(1);
  t = 1 && c;
  b = t & a;
  return 0;
}
