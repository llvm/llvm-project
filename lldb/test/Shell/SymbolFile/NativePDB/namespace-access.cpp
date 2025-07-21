// clang-format off
// REQUIRES: target-windows

// Test namespace lookup.
// RUN: %build --nodefaultlib -o %t.exe -- %s
// RUN: %lldb -f %t.exe -s \
// RUN:     %p/Inputs/namespace-access.lldbinit 2>&1 | FileCheck %s

struct S {
  char a[1];
};

namespace Outer {

  struct S {
    char a[2];
  };

  namespace Inner1 {
    struct S {
      char a[3];
    };

    namespace Inner2 {
      struct S {
        char a[4];
      };
    } // namespace Inner2
  } // namespace Inner1

  namespace Inner2 {
    struct S {
      char a[5];
    };
  } // namespace Inner2

  namespace {
    struct A {
      char a[6];
    };
  } // namespace

} // namespace Outer

namespace {
  struct A {
    char a[7];
  };
} // namespace

int main(int argc, char **argv) {
  S s;
  Outer::S os;
  Outer::Inner1::S oi1s;
  Outer::Inner1::Inner2::S oi1i2s;
  Outer::Inner2::S oi2s;
  A a1;
  Outer::A a2;
  return sizeof(s) + sizeof(os) + sizeof(oi1s) + sizeof(oi1i2s) + sizeof(oi2s) + sizeof(a1) + sizeof(a2);
}



// CHECK:      (lldb) type lookup S  
// CHECK:      struct S {
// CHECK:      struct S {
// CHECK:      struct S {
// CHECK:      struct S {
// CHECK:      struct S {
// CHECK:      }
// CHECK-NEXT: (lldb) type lookup ::S
// CHECK-NEXT: struct S {
// CHECK-NEXT:     char a[1];
// CHECK-NEXT: }
// CHECK-NEXT: (lldb) type lookup Outer::S
// CHECK-NEXT: struct S {
// CHECK-NEXT:     char a[2];
// CHECK-NEXT: }
// CHECK-NEXT: (lldb) type lookup Outer::Inner1::S
// CHECK-NEXT: struct S {
// CHECK-NEXT:     char a[3];
// CHECK-NEXT: }
// CHECK-NEXT: (lldb) type lookup Inner1::S
// CHECK-NEXT: struct S {
// CHECK-NEXT:     char a[3];
// CHECK-NEXT: }
// CHECK-NEXT: (lldb) type lookup Outer::Inner1::Inner2::S
// CHECK-NEXT: struct S {
// CHECK-NEXT:     char a[4];
// CHECK-NEXT: }
// CHECK-NEXT: (lldb) type lookup Inner2::S         
// CHECK-NEXT: struct S {
// CHECK:      struct S {
// CHECK:      }
// CHECK-NEXT: (lldb) type lookup Outer::Inner2::S        
// CHECK-NEXT: struct S {
// CHECK-NEXT:     char a[5];
// CHECK-NEXT: }
// CHECK-NEXT: (lldb) type lookup Outer::A        
// CHECK-NEXT: struct A {
// CHECK-NEXT:     char a[6];
// CHECK-NEXT: }
// CHECK-NEXT: (lldb) type lookup A       
// CHECK-NEXT: struct A {
// CHECK:      struct A {
// CHECK:      }
// CHECK-NEXT: (lldb) type lookup ::A
// CHECK-NEXT: struct A {
// CHECK-NEXT:     char a[7];
// CHECK-NEXT: }
// CHECK-NEXT: (lldb) expr sizeof(S) 
// CHECK-NEXT: (__size_t) $0 = 1
// CHECK-NEXT: (lldb) expr sizeof(A)
// CHECK-NEXT: (__size_t) $1 = 7
