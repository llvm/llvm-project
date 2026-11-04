// XFAIL: target-windows

// Tests that LLDB correctly parses global symbols
// starting with 'O'. On some platforms (e.g., Darwin)
// C-symbols are prefixed with a '_'. The LLDB Macho-O
// parses handles Objective-C metadata symbols starting
// with '_OBJC' specially. This test ensures that we don't
// lose track of regular global symbols with a '_O' prefix
// in this.

// RUN: %clang_host -c -g -fno-common %s -o %t.o
// RUN: %clang_host %t.o -o %t.out
// RUN: %lldb -b -x %t.out \
// RUN:       -o "b 29" \
// RUN:       -o "run" \
// RUN:       -o "p OglobalVar" \
// RUN:       -o "p Oabc" | FileCheck %s

typedef struct {
  int a;
} Oabc_t;

Oabc_t Oabc;
int OglobalVar;

int main(int argc, const char *argv[]) {
  Oabc.a = 15;
  OglobalVar = 10;
  return OglobalVar + Oabc.a;
}

// CHECK: (lldb) p OglobalVar
// CHECK: (int) 10
// CHECK: (lldb) p Oabc
// CHECK: (Oabc_t) (a = 15)
