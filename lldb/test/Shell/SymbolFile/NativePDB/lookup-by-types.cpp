// clang-format off

// RUN: %build -o %t.exe -- %s
// RUN: %lldb -f %t.exe -s \
// RUN:     %p/Inputs/lookup-by-types.lldbinit 2>&1 | FileCheck %s

class B;
class A {
public:
    static const A constA;
    static A a;
    static B b;
    int val = 1;
};
class B {
public:
    static A a;
    int val = 2;
};
A varA;
B varB;
const A A::constA = varA;
A A::a = varA;
B A::b = varB;
A B::a = varA;

int main(int argc, char **argv) {
  return varA.val + varB.val;
}

// CHECK:      image lookup -type A
// CHECK-NEXT: 1 match found in {{.*}}.exe
// CHECK-NEXT: compiler_type = "class A {
// CHECK-NEXT:     static const A constA;
// CHECK-NEXT:     static A a;
// CHECK-NEXT:     static B b;
// CHECK-NEXT: public:
// CHECK-NEXT:     int val;
// CHECK-NEXT: }"
// CHECK:      image lookup -type B
// CHECK-NEXT: 1 match found in {{.*}}.exe
// CHECK-NEXT:  compiler_type = "class B {
// CHECK-NEXT:     static A a;
// CHECK-NEXT: public:
// CHECK-NEXT:     int val;
// CHECK-NEXT: }"
