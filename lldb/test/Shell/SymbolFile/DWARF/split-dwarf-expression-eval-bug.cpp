// This tests a crash which occured under very specific circumstances. The
// interesting aspects of this test are:
// - we print a global variable from one compile unit
// - we are stopped in a member function of a class in a namespace
// - that namespace is also present in a third file, which also has a global
//   variable

// UNSUPPORTED: system-darwin, system-windows

// RUN: %clang_host -c -gsplit-dwarf -g %s -o %t1.o -DONE
// RUN: %clang_host -c -gsplit-dwarf -g %s -o %t2.o -DTWO
// RUN: %clang_host -c -gsplit-dwarf -g %s -o %t3.o -DTHREE
// RUN: %clang_host %t1.o %t2.o %t3.o -o %t
// RUN: %lldb %t -o "br set -n foo" -o run -o "expression bool_in_first_cu" -o exit \
// RUN:   | FileCheck %s

// CHECK: (lldb) expression bool_in_first_cu
// CHECK: (bool) $0 = true


#if defined(ONE)
bool bool_in_first_cu = true;
#elif defined(TWO)
bool bool_in_second_cu = true;

namespace NS {
void f() {}
}
#elif defined(THREE)
namespace NS {
struct S {
  void foo() {}
};
}

int main() { NS::S().foo(); }
#endif
