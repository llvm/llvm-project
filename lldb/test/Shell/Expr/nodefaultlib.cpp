// Test that we're able to evaluate expressions in inferiors without the
// standard library (and mmap-like functions in particular).

// REQUIRES: native
// XFAIL: system-linux && !(target-x86 || target-x86_64)
// XFAIL: system-netbsd || system-freebsd || system-darwin

// RUN: %build %s --nodefaultlib -o %t
// RUN: %lldb %t -o "b main" -o run -o "p call_me(5, 6)" -o exit \
// RUN:   | FileCheck %s

// CHECK: p call_me(5, 6)
// CHECK: (int) $0 = 30

int call_me(int x, long y) { return x * y; }

int main() { return call_me(4, 5); }
