// clang-format off
// REQUIRES: lld, x86

// RUN: %clang_cl --target=x86_64-windows-msvc -Od -GS- -std:c++20 -Z7 -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: lldb-test symbols --dump-ast %t.exe | FileCheck %s

class Foo {
public:
  void fun() {
    auto f = [this]() {
      int c = a;
      int d = b;
      return a + b;
    };
    f();
    int local = 42;
    auto g = [=]() mutable {
      return local + 1;
    };
    g();
  }

private:
  int a = 1;
  int b = 2;
};

int main() {
  Foo f;
  f.fun();
  return 0;
}

// CHECK:      namespace `public: void __cdecl Foo::fun(void)'::`1' {
// CHECK-NEXT:     class <lambda_1> {
// CHECK-NEXT:         int operator()() const;
// CHECK-NEXT:         Foo *__this;
// CHECK-NEXT:     };
// CHECK-NEXT:     class <lambda_2> {
// CHECK-NEXT:         int operator()();
// CHECK-NEXT:         int local;
// CHECK-NEXT:     };
// CHECK-NEXT: }

