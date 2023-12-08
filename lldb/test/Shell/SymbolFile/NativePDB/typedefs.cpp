// clang-format off

// REQUIRES: system-windows
// RUN: %build --compiler=clang-cl --nodefaultlib -o %t.exe -- %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 lldb-test symbols -dump-ast %t.exe | FileCheck %s

namespace A {
  namespace B {
    using NamespaceTypedef = double;
  }
  template<typename T>
  class C {
  public:
    using ClassTypedef = T;
  };
  using ClassTypedef = C<char>::ClassTypedef;
  using ClassTypedef2 = C<wchar_t>::ClassTypedef;
  
  template<typename T>
  using AliasTemplate = typename C<T>::ClassTypedef;
}

namespace {
  using AnonNamespaceTypedef = bool;
}

using IntTypedef = int;

using ULongArrayTypedef = unsigned long[10];

using RefTypedef = long double*&;

using FuncPtrTypedef = long long(*)(int&, unsigned char**, short[], const double, volatile bool);

using VarArgsFuncTypedef = char(*)(void*, long, unsigned short, unsigned int, ...);

using VarArgsFuncTypedefA = float(*)(...);

int main(int argc, char **argv) {
  long double *Ptr;
  
  A::B::NamespaceTypedef *X0;
  A::C<char>::ClassTypedef *X1;
  A::C<wchar_t>::ClassTypedef *X2;
  AnonNamespaceTypedef *X3;
  IntTypedef *X4;
  ULongArrayTypedef *X5;
  RefTypedef X6 = Ptr;
  FuncPtrTypedef X7;
  VarArgsFuncTypedef X8;
  VarArgsFuncTypedefA X9;
  A::AliasTemplate<float> X10;
  return 0;
}


// CHECK:      namespace  {
// CHECK-NEXT:     typedef bool AnonNamespaceTypedef;
// CHECK-NEXT: }
// CHECK-NEXT: typedef unsigned long ULongArrayTypedef[10];
// CHECK-NEXT: typedef double *&RefTypedef;
// CHECK-NEXT: namespace A {
// CHECK-NEXT:     namespace B {
// CHECK-NEXT:         typedef double NamespaceTypedef;
// CHECK-NEXT:     }
// CHECK-NEXT:     typedef float AliasTemplate<float>;
// CHECK-NEXT: }
// CHECK-NEXT: typedef long long (*FuncPtrTypedef)(int &, unsigned char **, short *, const double, volatile bool);
// CHECK-NEXT: typedef char (*VarArgsFuncTypedef)(void *, long, unsigned short, unsigned int, ...);
// CHECK-NEXT: typedef float (*VarArgsFuncTypedefA)(...);
// CHECK-NEXT: typedef int IntTypedef;
