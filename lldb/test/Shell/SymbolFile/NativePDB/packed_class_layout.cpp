// clang-format off
// REQUIRES: lld, x86

// Make sure class layout is correct.
// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe \
// RUN:   -o "expr a" -o "expr b.c" -o "expr b.u.c" -o "expr b.u.i" -o "exit" | FileCheck %s

// CHECK:      (lldb) expr a
// CHECK-NEXT: (A) $0 = (d1 = 'a', d2 = 1, d3 = 2, d4 = 'b')
// CHECK-NEXT: (lldb) expr b.c
// CHECK-NEXT: (char) $1 = 'a'
// CHECK-NEXT: (lldb) expr b.u.c
// CHECK-NEXT: (char[2]) $2 = "b"
// CHECK-NEXT: (lldb) expr b.u.i
// CHECK-NEXT: (int) $3 = 98

struct __attribute__((packed, aligned(1))) A {
  char d1;
  int d2;
  int d3;
  char d4;
};

struct __attribute__((packed, aligned(1))) B {
  char c;
  union {
    char c[2];
    int i;
  } u;
};

A a = {'a', 1, 2, 'b'};
B b = {'a', {"b"}};

int main() {
  return 0;
}
