// REQUIRES: lld

// Test that simple types can be found
// RUN: %build --std=c++20 --nodefaultlib --compiler=clang-cl --arch=64 -o %t.exe -- %s
// RUN: lldb-test symbols %t.exe | FileCheck %s
// RUN: lldb-test symbols %t.exe | FileCheck --check-prefix=FUNC-PARAMS %s

bool *PB;
bool &RB = *PB;
bool *&RPB = PB;
const bool &CRB = RB;
bool *const BC = 0;
const bool *const CBC = 0;

long AL[2];

const volatile short CVS = 0;
const short CS = 0;
volatile short VS;

struct ReturnedStruct1 {};
struct ReturnedStruct2 {};

struct MyStruct {
  static ReturnedStruct1 static_fn(char *) { return {}; }
  ReturnedStruct2 const_member_fn(char *) const { return {}; }
  void volatile_member_fn() volatile {};
  void member_fn() {};
};

void (*PF)(int, bool *, const float, double, ...);

using Func = void(char16_t, MyStruct &);
Func *PF2;

using SomeTypedef = long;
SomeTypedef ST;

int main() {
  bool b;
  char c;
  unsigned char uc;
  char8_t c8;

  short s;
  unsigned short us;
  wchar_t wc;
  char16_t c16;

  int i;
  unsigned int ui;
  long l;
  unsigned long ul;
  char32_t c32;

  long long ll;
  unsigned long long ull;

  MyStruct my_struct;

  _Float16 f16;

  _Complex float cf;
  _Complex double cd;

  __int128 i128;
  unsigned __int128 ui128;

  decltype(nullptr) np;
}

// CHECK-DAG: Type{{.*}} , name = "decltype(nullptr)", compiler_type = 0x{{[0-9a-f]+}} nullptr_t

// CHECK-DAG: Type{{.*}} , name = "bool", size = 1, compiler_type = 0x{{[0-9a-f]+}} _Bool
// CHECK-DAG: Type{{.*}} , name = "char", size = 1, compiler_type = 0x{{[0-9a-f]+}} char
// CHECK-DAG: Type{{.*}} , name = "unsigned char", size = 1, compiler_type = 0x{{[0-9a-f]+}} unsigned char
// CHECK-DAG: Type{{.*}} , name = "char8_t", size = 1, compiler_type = 0x{{[0-9a-f]+}} char8_t

// CHECK-DAG: Type{{.*}} , name = "short", size = 2, compiler_type = 0x{{[0-9a-f]+}} short
// CHECK-DAG: Type{{.*}} , name = "const volatile short", size = 2, compiler_type = 0x{{[0-9a-f]+}} const volatile short
// CHECK-DAG: Type{{.*}} , name = "const short", size = 2, compiler_type = 0x{{[0-9a-f]+}} const short
// CHECK-DAG: Type{{.*}} , name = "volatile short", size = 2, compiler_type = 0x{{[0-9a-f]+}} volatile short

// CHECK-DAG: Type{{.*}} , name = "unsigned short", size = 2, compiler_type = 0x{{[0-9a-f]+}} unsigned short
// CHECK-DAG: Type{{.*}} , name = "wchar_t", size = 2, compiler_type = 0x{{[0-9a-f]+}} wchar_t
// CHECK-DAG: Type{{.*}} , name = "char16_t", size = 2, compiler_type = 0x{{[0-9a-f]+}} char16_t

// CHECK-DAG: Type{{.*}} , name = "int", size = 4, compiler_type = 0x{{[0-9a-f]+}} int
// CHECK-DAG: Type{{.*}} , name = "unsigned", size = 4, compiler_type = 0x{{[0-9a-f]+}} unsigned int
// CHECK-DAG: Type{{.*}} , name = "long", size = 4, compiler_type = 0x{{[0-9a-f]+}} long
// CHECK-DAG: Type{{.*}} , name = "unsigned long", size = 4, compiler_type = 0x{{[0-9a-f]+}} unsigned long
// CHECK-DAG: Type{{.*}} , name = "char32_t", size = 4, compiler_type = 0x{{[0-9a-f]+}} char32_t

// CHECK-DAG: Type{{.*}} , name = "long long", size = 8, compiler_type = 0x{{[0-9a-f]+}} long long
// CHECK-DAG: Type{{.*}} , name = "unsigned long long", size = 8, compiler_type = 0x{{[0-9a-f]+}} unsigned long long

// CHECK-DAG: Type{{.*}} , name = "__int128", size = 16, compiler_type = 0x{{[0-9a-f]+}} __int128
// CHECK-DAG: Type{{.*}} , name = "unsigned __int128", size = 16, compiler_type = 0x{{[0-9a-f]+}} unsigned __int128

// CHECK-DAG: Type{{.*}} , name = "_Float16", size = 2, compiler_type = 0x{{[0-9a-f]+}} __fp16
// CHECK-DAG: Type{{.*}} , name = "float", size = 4, compiler_type = 0x{{[0-9a-f]+}} float
// CHECK-DAG: Type{{.*}} , name = "const float", size = 4, compiler_type = 0x{{[0-9a-f]+}} const float

// CHECK-DAG: Type{{.*}} , name = "_Complex float", size = 4, compiler_type = 0x{{[0-9a-f]+}} _Complex float
// CHECK-DAG: Type{{.*}} , name = "_Complex double", size = 8, compiler_type = 0x{{[0-9a-f]+}} _Complex double

// CHECK-DAG:  Type{{.*}} , name = "ReturnedStruct1", size = 1, decl = simple-types.cpp:21, compiler_type = 0x{{[0-9a-f]+}} struct ReturnedStruct1 {
// CHECK-DAG:  Type{{.*}} , name = "ReturnedStruct2", size = 1, decl = simple-types.cpp:22, compiler_type = 0x{{[0-9a-f]+}} struct ReturnedStruct2 {
// CHECK-DAG:  Type{{.*}} , name = "MyStruct", size = 1, decl = simple-types.cpp:24, compiler_type = 0x{{[0-9a-f]+}} struct MyStruct {

// CHECK-DAG:  Type{{.*}} , size = 8, compiler_type = 0x{{[0-9a-f]+}} struct MyStruct *const
// CHECK-DAG:  Type{{.*}} , size = 8, compiler_type = 0x{{[0-9a-f]+}} const struct MyStruct *const
// CHECK-DAG:  Type{{.*}} , size = 8, compiler_type = 0x{{[0-9a-f]+}} volatile struct MyStruct *const
// CHECK-DAG:  Type{{.*}} , size = 8, compiler_type = 0x{{[0-9a-f]+}} struct MyStruct &

// CHECK-DAG: Type{{.*}} , name = "const bool", size = 1, compiler_type = 0x{{[0-9a-f]+}} const _Bool
// CHECK-DAG: Type{{.*}} , size = 8, compiler_type = 0x{{[0-9a-f]+}} _Bool &
// CHECK-DAG: Type{{.*}} , size = 8, compiler_type = 0x{{[0-9a-f]+}} _Bool *
// CHECK-DAG: Type{{.*}} , size = 8, compiler_type = 0x{{[0-9a-f]+}} _Bool *&
// CHECK-DAG: Type{{.*}} , size = 8, compiler_type = 0x{{[0-9a-f]+}} const _Bool &
// CHECK-DAG: Type{{.*}} , size = 8, compiler_type = 0x{{[0-9a-f]+}} _Bool *const
// CHECK-DAG: Type{{.*}} , size = 8, compiler_type = 0x{{[0-9a-f]+}} const _Bool *const

// CHECK-DAG: Type{{.*}} , name = "SomeTypedef", size = 4, compiler_type = 0x{{[0-9a-f]+}} typedef SomeTypedef
// CHECK-DAG: Type{{.*}} , name = "Func", size = 0, compiler_type = 0x{{[0-9a-f]+}} typedef Func

// CHECK-DAG: Type{{.*}} , size = 0, compiler_type = 0x{{[0-9a-f]+}} int (void)
// CHECK-DAG: Type{{.*}} , size = 0, compiler_type = 0x{{[0-9a-f]+}} void (void)

// CHECK-DAG: Type{{.*}} , size = 0, compiler_type = 0x{{[0-9a-f]+}} void (int, _Bool *, const float, double, ...)
// CHECK-DAG: Type{{.*}} , size = 8, compiler_type = 0x{{[0-9a-f]+}} void (*)(int, _Bool *, const float, double, ...)

// CHECK-DAG: Type{{.*}} , size = 0, compiler_type = 0x{{[0-9a-f]+}} void (char16_t, struct MyStruct &)
// CHECK-DAG: Type{{.*}} , size = 8, compiler_type = 0x{{[0-9a-f]+}} void (*)(char16_t, struct MyStruct &)

// CHECK-DAG: Type{{.*}} , size = 0, compiler_type = 0x{{[0-9a-f]+}} struct ReturnedStruct1 (char *)
// CHECK-DAG: Type{{.*}} , size = 0, compiler_type = 0x{{[0-9a-f]+}} struct ReturnedStruct2 (char *)

// CHECK-DAG: Type{{.*}} , size = 8, compiler_type = 0x{{[0-9a-f]+}} long[2]

// double is used as a parameter to `PF`, but not created as an LLDB type
// FUNC-PARAMS-NOT: Type{{.*}} , name = "double"
