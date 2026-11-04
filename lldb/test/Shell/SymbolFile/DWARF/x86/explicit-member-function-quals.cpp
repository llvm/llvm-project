// XFAIL: *
//
// FIXME: Explicit object parameter is not shown in
// type lookup output. This is because we don't attach
// valid source locations to decls in the DWARF AST,
// so the ParmVarDecl::isExplicitObjectParameter fails.

// Tests that we correctly deduce the CV-quals and storage
// class of explicit object member functions.
//
// RUN: %clangxx_host %s -glldb -target x86_64-pc-linux -g -std=c++23 -c -o %t
// RUN: %lldb %t -b -o "type lookup Foo" 2>&1 | FileCheck %s
//
// CHECK:      (lldb) type lookup Foo
// CHECK-NEXT: struct Foo {
// CHECK-NEXT:      void Method(this Foo);
// CHECK-NEXT:      void cMethod(this const Foo &) const;
// CHECK-NEXT:      void vMethod(this volatile Foo &) volatile;
// CHECK-NEXT:      void cvMethod(this const volatile Foo &) const volatile;
// CHECK-NEXT: }

struct Foo {
  void Method(this Foo) {}
  void cMethod(this Foo const &) {}
  void vMethod(this Foo volatile &) {}
  void cvMethod(this Foo const volatile &) {}
} f;
