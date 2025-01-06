// clang-format off
// REQUIRES: lld, x86

// Make sure class layout is correct.
// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: %lldb -f %t.exe -s \
// RUN:     %p/Inputs/class_layout.lldbinit 2>&1 | FileCheck %s

// CHECK:      (lldb) expr a
// CHECK-NEXT: (A) $0 = (d1 = 'a', d2 = 1, d3 = 2, d4 = 'b')
// CHECK-NEXT: (lldb) expr b.c
// CHECK-NEXT: (char) $1 = 'a'
// CHECK-NEXT: (lldb) expr b.u.c
// CHECK-NEXT: (char[2]) $2 = "b"
// CHECK-NEXT: (lldb) expr b.u.i
// CHECK-NEXT: (int) $3 = 98
// CHECK-NEXT: (lldb) expr c
// CHECK-NEXT: (C) $4 = {
// CHECK-NEXT:   c = 'a'
// CHECK-NEXT:   x = 65
// CHECK-NEXT:    = {
// CHECK-NEXT:      = {
// CHECK-NEXT:       c1 = 'b'
// CHECK-NEXT:        = {
// CHECK-NEXT:          = {
// CHECK-NEXT:           s3 = {
// CHECK-NEXT:             x = ([0] = 66, [1] = 67, [2] = 68)
// CHECK-NEXT:           }
// CHECK-NEXT:           c3 = 'c'
// CHECK-NEXT:         }
// CHECK-NEXT:          = {
// CHECK-NEXT:           c4 = 'B'
// CHECK-NEXT:           s4 = {
// CHECK-NEXT:             x = ([0] = 67, [1] = 68, [2] = 99)
// CHECK-NEXT:           }
// CHECK-NEXT:           s1 = {
// CHECK-NEXT:             x = ([0] = 69, [1] = 70, [2] = 71)
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:      = {
// CHECK-NEXT:       s2 = {
// CHECK-NEXT:         x = ([0] = 98, [1] = 66, [2] = 67)
// CHECK-NEXT:       }
// CHECK-NEXT:       c2 = 'D'
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: (lldb) type lookup C
// CHECK-NEXT: struct C {
// CHECK-NEXT:     char c;
// CHECK-NEXT:     int x;
// CHECK-NEXT:     union {
// CHECK-NEXT:         struct {
// CHECK-NEXT:             char c1;
// CHECK-NEXT:             union {
// CHECK-NEXT:                 struct {
// CHECK-NEXT:                     S3 s3;
// CHECK-NEXT:                     char c3;
// CHECK-NEXT:                 };
// CHECK-NEXT:                 struct {
// CHECK-NEXT:                     char c4;
// CHECK-NEXT:                     S3 s4;
// CHECK-NEXT:                     S3 s1;
// CHECK-NEXT:                 };
// CHECK-NEXT:             };
// CHECK-NEXT:         };
// CHECK-NEXT:         struct {
// CHECK-NEXT:             S3 s2;
// CHECK-NEXT:             char c2;
// CHECK-NEXT:         };
// CHECK-NEXT:     };
// CHECK-NEXT: }



// Test packed stuct layout.
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

// Test struct layout with anonymous union/struct.
struct S3 {
  short x[3];
};

struct C {
  char c;
  int x;
  union {
    struct {
      char c1;
      union {
        struct {
          S3 s3;
          char c3;
        };
        struct {
          char c4;
          S3 s4;
        };
      };
      S3 s1;
    };
    struct {
      S3 s2;
      char c2;
    };
  };
};

A a = {'a', 1, 2, 'b'};
B b = {'a', {"b"}};
C c = {.c = 'a', .x = 65, .c1 = 'b', .s3 = {66, 67, 68}, .c3 = 'c', .s1={69, 70, 71}};

int main() {
  return 0;
}
