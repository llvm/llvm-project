// clang-format off
// REQUIRES: msvc, target-windows

// Test that breakpoints on functions are placed at the end of the prologue
// that the procedure records describe (DbgStart), also when the body is on the
// same line as the start of the function.

// RUN: %build --compiler=msvc --nodefaultlib -o %t.exe -- %s
// RUN: %lldb -f %t.exe -b -o "break set -n Struct::get" -o "break set -n add" \
// RUN:     -o run -o "frame variable this->n" -o continue 2>&1 | FileCheck %s

struct Struct {
  int n = 30;
  int get() { return n; }
  // CHECK: Breakpoint 1: where = {{.*}}`{{.*}}Struct::get{{.*}} + {{[1-9][0-9]*}} at prologue-end-msvc.cpp:[[@LINE-1]]
};

int add(int a, int b) {
  int c = a + b;
  // CHECK: Breakpoint 2: where = {{.*}}`{{.*}}add{{.*}} + {{[1-9][0-9]*}} at prologue-end-msvc.cpp:[[@LINE-1]]
  return c;
}

int main() {
  Struct s;
  int r = s.get();
  return r + add(1, 2);
}

// CHECK: stop reason = breakpoint 1.1
// CHECK: (lldb) frame variable this->n
// CHECK-NEXT: (int) this->n = 30
// CHECK: stop reason = breakpoint 2.1
