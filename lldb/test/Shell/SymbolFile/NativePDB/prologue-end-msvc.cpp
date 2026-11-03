// clang-format off
// REQUIRES: msvc

// Test that breakpoints on functions are placed at the end of the prologue
// that the procedure records describe (DbgStart), also when the body is on the
// same line as the start of the function.

// RUN: %build --compiler=msvc --nodefaultlib -o %t.exe -- %s
// RUN: %lldb -f %t.exe -b -o "break set -n add" -o "break set -n Struct::get" \
// RUN:     2>&1 | FileCheck %s

struct Struct {
  int n = 30;
  int get() { return n; }
};

int add(int a, int b) {
  int c = a + b;
  return c;
}

int main() {
  Struct s;
  return s.get() + add(1, 2);
}

// CHECK: Breakpoint 1: where = {{.*}}`{{.*}}add{{.*}} + {{[1-9][0-9]*}} at prologue-end-msvc.cpp:18
// CHECK: Breakpoint 2: where = {{.*}}`{{.*}}Struct::get{{.*}} + {{[1-9][0-9]*}} at prologue-end-msvc.cpp:14
