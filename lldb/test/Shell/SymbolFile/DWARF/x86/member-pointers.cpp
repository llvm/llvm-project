// REQUIRES: lld

// Itanium ABI:
// RUN: %clang --target=x86_64-pc-linux -gdwarf -c -o %t_linux.o %s
// RUN: %lldb -f %t_linux.o -b -o "target variable s1 s2 m1 m2 v1 v2 v3 v4" | FileCheck --check-prefix=CHECK-GNU %s
//
// CHECK-GNU: (void (Single1::*)()) s1 = 0x00000000000000000000000000000000
// CHECK-GNU: (void (Single2::*)()) s2 = 0x00000000000000000000000000000000
// CHECK-GNU: (void (Multiple1::*)()) m1 = 0x00000000000000000000000000000000
// CHECK-GNU: (void (Multiple2::*)()) m2 = 0x00000000000000000000000000000000
// CHECK-GNU: (void (Virtual1::*)()) v1 = 0x00000000000000000000000000000000
// CHECK-GNU: (void (Virtual2::*)()) v2 = 0x00000000000000000000000000000000
// CHECK-GNU: (void (Virtual3::*)()) v3 = 0x00000000000000000000000000000000
// CHECK-GNU: (void (Virtual4::*)()) v4 = 0x00000000000000000000000000000000

// Microsoft ABI:
// RUN: %clang_cl --target=x86_64-windows-msvc -c -gdwarf -o %t_win.obj /GS- -- %s
// RUN: lld-link /out:%t_win.exe %t_win.obj /entry:main /debug /nodefaultlib
// RUN: %lldb -f %t_win.exe -b -o "target variable s1 s2 m1 m2 v1 v2 v3 v4" | FileCheck --check-prefix=CHECK-MSVC %s
//
// CHECK-MSVC: (void (Single1::*)()) s1 = 0x0000000000000000
// CHECK-MSVC: (void (Single2::*)()) s2 = 0x0000000000000000
// CHECK-MSVC: (void (Multiple1::*)()) m1 = 0x00000000000000000000000000000000
// CHECK-MSVC: (void (Multiple2::*)()) m2 = 0x00000000000000000000000000000000
// CHECK-MSVC: (void (Virtual1::*)()) v1 = 0xffffffff000000000000000000000000
// CHECK-MSVC: (void (Virtual2::*)()) v2 = 0xffffffff000000000000000000000000
// CHECK-MSVC: (void (Virtual3::*)()) v3 = 0xffffffff000000000000000000000000
// CHECK-MSVC: (void (Virtual4::*)()) v4 = 0xffffffff000000000000000000000000

// clang-format off
struct Single1 { void s1() {} };
struct Single2 : Single1 { void s2() {} };

struct Helper {};
struct Multiple1 : Single1, Helper { void m1() {} };
struct Multiple2 : Multiple1 { void m2() {} };

struct Virtual1 : virtual Single1 { void v1() {} };
struct Virtual2 : Virtual1 { void v2() {} };
struct Virtual3 : virtual Multiple1 { void v3() {} };
struct Virtual4 : Virtual1, Helper { void v4() {} };

void (Single1::*s1)() = nullptr;
void (Single2::*s2)() = nullptr;
void (Multiple1::*m1)() = nullptr;
void (Multiple2::*m2)() = nullptr;
void (Virtual1::*v1)() = nullptr;
void (Virtual2::*v2)() = nullptr;
void (Virtual3::*v3)() = nullptr;
void (Virtual4::*v4)() = nullptr;

int main(int argc, char *argv[]) {
  // We need to force emission of type info to DWARF. That's reasonable, because
  // Clang doesn't know that we need it to infer member-pointer sizes. We could
  // probably teach Clang to do so, but in most real-world scenarios this might
  // be a non-issue.
  Virtual1 vi1;
  Virtual2 vi2;
  Virtual3 vi3;
  Virtual4 vi4;
  int sum = sizeof(Single2) + sizeof(Multiple2);
  return argc < sum ? 0 : 1;
}
