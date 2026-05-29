// clang-format off
// REQUIRES: msvc

// RUN: %build --compiler=msvc --nodefaultlib --std c++20 -o %t.exe -- %s
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

// CHECK:      namespace `public: void __cdecl Foo::fun(void)'::`{{.*}}' {
// CHECK-NEXT:     class <lambda_1> {
// CHECK:              int operator()() const;
// CHECK:              Foo *__this;
// CHECK-NEXT:     };
// CHECK-NEXT:     class <lambda_2> {
// CHECK:              int operator()();
// CHECK:              int local;
// CHECK-NEXT:     };
// CHECK-NEXT: }

