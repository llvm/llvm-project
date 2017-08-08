// Compile with "cl /c /Zi /GR- symbolformat.cpp"
// Compile symbolformat-fpo.cpp (see file for instructions)
// Link with "link symbolformat.obj symbolformat-fpo.obj /debug /nodefaultlib
//    /entry:main /out:symbolformat.exe"

int __cdecl _purecall(void) { return 0; }

enum TestEnum {
  Value,
  Value10 = 10
};

enum class TestEnumClass {
  Value,
  Value10 = 10
};

struct A {
  virtual void PureFunc() = 0 {}
  virtual void VirtualFunc() {}
  void RegularFunc() {}
};

struct VirtualBase {
};

struct B : public A, protected virtual VirtualBase {
  void PureFunc() override {}
};

struct MemberTest {
  enum NestedEnum {
    FirstVal,
    SecondVal
  };

  typedef int NestedTypedef;

  NestedEnum m_nested_enum;
  NestedTypedef m_typedef;
  bool m_bool;
  char m_char;
  wchar_t m_wchar_t;
  int m_int;
  unsigned m_unsigned;
  long m_long;
  unsigned long m_unsigned_long;
  __int64 m_int64;
  unsigned __int64 m_unsigned_int64;
  float m_float;
  double m_double;
  void (*m_pfn_2_args)(int, double);
  int m_multidimensional_array[2][3];
};

typedef int IntType;
typedef A ClassAType;

int g_global_int;
void *g_global_pointer = nullptr;

typedef int int_array[3];
int_array g_array = { 1, 2, 3 };
int_array *g_pointer_to_array = &g_array;
const int *g_pointer_to_const_int = nullptr;
int * const g_const_pointer_to_int = nullptr;
const int * const g_const_pointer_to_const_int = nullptr;

int main(int argc, char **argv) {
  // Force symbol references so the linker generates debug info
  B b;
  MemberTest members;
  auto PureAddr = &B::PureFunc;
  auto VirtualAddr = &A::PureFunc;
  auto RegularAddr = &A::RegularFunc;
  TestEnum Enum = Value;
  TestEnumClass EnumClass = TestEnumClass::Value10;
  IntType Int = 12;
  ClassAType *ClassA = &b;
  return 0;
}
