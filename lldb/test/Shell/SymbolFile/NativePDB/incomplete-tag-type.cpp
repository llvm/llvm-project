// clang-format off
// REQUIRES: lld, x86

// RUN: %clang_cl --target=x86_64-windows-msvc -c /Fo%t1.obj -- %p/Inputs/incomplete-tag-type.cpp
// RUN: %clang_cl --target=x86_64-windows-msvc /O1 /Z7 -c /Fo%t2.obj -- %s
// RUN: lld-link /debug:full /nodefaultlib /entry:main %t1.obj %t2.obj /out:%t.exe /pdb:%t.pdb
// RUN: %lldb -f %t.exe -o \
// RUN:   "settings set interpreter.stop-command-source-on-error false" \
// RUN:   -o "expression b" -o "expression d" -o "expression static_e_ref" -o "exit" 2>&1 | FileCheck %s

// CHECK: (lldb) expression b
// CHECK: (B) $0 = {}
// CHECK: (lldb) expression d
// CHECK: (D) $1 = {}
// CHECK: (lldb) expression static_e_ref
// CHECK: error:{{.*}}incomplete type 'E' where a complete type is required

// Complete base class.
struct A { int x; A(); };
struct B : A {};
B b;

// Complete data member.
struct C {
  C();
};

struct D {
  C c;
};
D d;

// Incomplete static data member should return error.
struct E {
  E();
};

struct F {
  static E static_e;
};

E F::static_e = E();
E& static_e_ref = F::static_e;

int main(){}
