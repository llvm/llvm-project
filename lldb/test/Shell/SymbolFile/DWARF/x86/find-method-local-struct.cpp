// RUN: %clang %s -g -c -o %t --target=x86_64-apple-macosx
// RUN: lldb-test symbols --name=foo --find=function --function-flags=method %t | \
// RUN:   FileCheck %s

// CHECK-DAG: name = "sbar::foo()", mangled = "_ZN4sbar3fooEv"
// CHECK-DAG: name = "ffbar()::sbar::foo()", mangled = "_ZZ5ffbarvEN4sbar3fooEv"

struct sbar {
  void foo();
};
void sbar::foo() {}

void ffbar() {
  struct sbar {
    void foo() {}
  };
  sbar a;
  a.foo();
}
