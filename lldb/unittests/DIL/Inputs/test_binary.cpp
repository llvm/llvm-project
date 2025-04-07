//===-- test_binary.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

static void TestArithmetic() {
  char c = 10;
  unsigned char uc = 1;
  int a = 1;
  int int_max = std::numeric_limits<int>::max();
  int int_min = std::numeric_limits<int>::min();
  unsigned int uint_max = std::numeric_limits<unsigned int>::max();
  unsigned int uint_zero = 0;
  long long ll_max = std::numeric_limits<long long>::max();
  long long ll_min = std::numeric_limits<long long>::min();
  unsigned long long ull_max = std::numeric_limits<unsigned long long>::max();
  unsigned long long ull_zero = 0;

  int x = 2;
  int &r = x;
  int *p = &x;

  typedef int &myr;
  myr my_r = x;

  auto fnan = std::numeric_limits<float>::quiet_NaN();
  auto fsnan = std::numeric_limits<float>::signaling_NaN();
  // Smallest positive non-zero float denormal
  auto fdenorm = 0x0.1p-145f;

  // BREAK(TestArithmetic)
  // BREAK(TestZeroDivision)
}

static void TestBitwiseOperators() {
  bool var_true = true;
  bool var_false = false;

  unsigned long long ull_max = std::numeric_limits<unsigned long long>::max();
  unsigned long long ull_zero = 0;

  struct S {
  } s;

  const char *p = nullptr;

  uint32_t mask_ff = 0xFF;

  // BREAK(TestBitwiseOperators)
}

static void TestPointerArithmetic() {
  int *p_null = nullptr;
  const char *p_char1 = "hello";

  typedef const char *my_char_ptr;
  my_char_ptr my_p_char1 = p_char1;

  int offset = 5;
  int array[10];
  array[0] = 0;
  array[offset] = offset;

  int(&array_ref)[10] = array;

  int *p_int0 = &array[0];
  int **pp_int0 = &p_int0;
  const int *cp_int0 = &array[0];
  const int *cp_int5 = &array[offset];
  const int *&rcp_int0 = cp_int0;

  typedef int *td_int_ptr_t;
  td_int_ptr_t td_int_ptr0 = &array[0];

  void *p_void = (void *)p_char1;
  void **pp_void0 = &p_void;
  void **pp_void1 = pp_void0 + 1;

  std::nullptr_t std_nullptr_t = nullptr;

  // BREAK(TestPointerArithmetic)
  // BREAK(PointerPointerArithmeticFloat)
  // BREAK(PointerPointerComparison)
  // BREAK(PointerIntegerComparison)
  // BREAK(TestPointerDereference)
}

static void TestLogicalOperators() {
  bool trueVar = true;
  bool falseVar = false;

  const char *p_ptr = "🦊";
  const char *p_nullptr = nullptr;

  int array[2] = {1, 2};

  struct S {
  } s;

  // BREAK(TestLogicalOperators)
}

static void TestLocalVariables() {
  int a = 1;
  int b = 2;

  char c = -3;
  unsigned short s = 4;

  // BREAK(TestLocalVariables)
}

static void TestMemberOf() {
  int x = 2;
  struct Sx {
    int x;
    int &r;
    char y;
  } s{1, x, 2};

  Sx &sr = s;
  Sx *sp = &s;

  Sx sarr[2] = {{5, x, 2}, {1, x, 3}};

  using SxAlias = Sx;
  SxAlias sa{3, x, 4};

  // BREAK(TestMemberOf)
}

static void TestMemberOfInheritance() {
  struct A {
    int a_;
  } a{1};

  struct B {
    int b_;
  } b{2};

  struct C : A, B {
    int c_;
  } c{{1}, {2}, 3};

  struct D : C {
    int d_;
    A fa_;
  } d{{{1}, {2}, 3}, 4, {5}};

  // Virtual inheritance example.
  struct Animal {
    virtual ~Animal() = default;
    int weight_;
  };
  struct Mammal : virtual Animal {};
  struct WingedAnimal : virtual Animal {};
  struct Bat : Mammal, WingedAnimal {
  } bat;
  bat.weight_ = 10;

  // Empty bases example.
  struct IPlugin {
    virtual ~IPlugin() {}
  };
  struct Plugin : public IPlugin {
    int x;
    int y;
  };
  Plugin plugin;
  plugin.x = 1;
  plugin.y = 2;

  struct ObjectBase {
    int x;
  };
  struct Object : ObjectBase {};
  struct Engine : Object {
    int y;
    int z;
  };

  Engine engine;
  engine.x = 1;
  engine.y = 2;
  engine.z = 3;

  // Empty multiple inheritance with empty base.
  struct Base {
    int x;
    int y;
    virtual void Do() = 0;
    virtual ~Base() {}
  };
  struct Mixin {};
  struct Parent : private Mixin, public Base {
    int z;
    virtual void Do() {};
  };
  Parent obj;
  obj.x = 1;
  obj.y = 2;
  obj.z = 3;
  Base *parent_base = &obj;
  Parent *parent = &obj;

  // BREAK(TestMemberOfInheritance)
}

static void TestMemberOfAnonymousMember() {
  struct A {
    struct {
      int x = 1;
    };
    int y = 2;
  } a;

  struct B {
    // Anonymous struct inherits another struct.
    struct : public A {
      int z = 3;
    };
    int w = 4;
    A a;
  } b;

  // Anonymous classes and unions.
  struct C {
    union {
      int x = 5;
    };
    class {
    public:
      int y = 6;
    };
  } c;

  // Multiple levels of anonymous structs.
  struct D {
    struct {
      struct {
        int x = 7;
        struct {
          int y = 8;
        };
      };
      int z = 9;
      struct {
        int w = 10;
      };
    };
  } d;

  struct E {
    struct IsNotAnon {
      int x = 11;
    };
  } e;

  struct F {
    struct {
      int x = 12;
    } named_field;
  } f;

  // Inherited unnamed struct without an enclosing parent class.
  struct : public A {
    struct {
      int z = 13;
    };
  } unnamed_derived;

  struct DerivedB : public B {
    struct {
      // `w` in anonymous struct overrides `w` from `B`.
      int w = 14;
      int k = 15;
    };
  } derb;

  // BREAK(TestMemberOfAnonymousMember)
}

static void TestIndirection() {
  int val = 1;
  int *p = &val;

  typedef int *myp;
  myp my_p = &val;

  typedef int *&mypr;
  mypr my_pr = p;

  // BREAK(TestIndirection)
}

// Referenced by TestInstanceVariables
class C {
public:
  int field_ = 1337;
};

// Referenced by TestAddressOf
int globalVar = 0xDEADBEEF;
extern int externGlobalVar;

int *globalPtr = &globalVar;
int &globalRef = globalVar;

namespace ns {
int globalVar = 13;
int *globalPtr = &globalVar;
int &globalRef = globalVar;
} // namespace ns

void TestGlobalVariableLookup() {
  // BREAK(TestGlobalVariableLookup)
}

class TestMethods {
public:
  void TestInstanceVariables() {
    C c;
    c.field_ = -1;

    C &c_ref = c;
    C *c_ptr = &c;

    // BREAK(TestInstanceVariables)
  }

  void TestAddressOf(int param) {
    int x = 42;
    int &r = x;
    int *p = &x;
    int *&pr = p;

    typedef int *&mypr;
    mypr my_pr = p;

    std::string s = "hello";
    const char *s_str = s.c_str();

    char c = 1;

    // BREAK(TestAddressOf)
  }

private:
  int field_ = 1;
};

static void TestSubscript() {
  const char *char_ptr = "lorem";
  const char char_arr[] = "ipsum";

  int int_arr[] = {1, 2, 3};

  C c_arr[2];
  c_arr[0].field_ = 0;
  c_arr[1].field_ = 1;

  C(&c_arr_ref)[2] = c_arr;

  int idx_1 = 1;
  const int &idx_1_ref = idx_1;

  typedef int td_int_t;
  typedef td_int_t td_td_int_t;
  typedef int *td_int_ptr_t;
  typedef int &td_int_ref_t;

  td_int_t td_int_idx_1 = 1;
  td_td_int_t td_td_int_idx_2 = 2;

  td_int_t td_int_arr[3] = {1, 2, 3};
  td_int_ptr_t td_int_ptr = td_int_arr;

  td_int_ref_t td_int_idx_1_ref = td_int_idx_1;
  td_int_t(&td_int_arr_ref)[3] = td_int_arr;

  unsigned char uchar_idx = std::numeric_limits<unsigned char>::max();
  uint8_t uint8_arr[256];
  uint8_arr[255] = 0xAB;
  uint8_t *uint8_ptr = uint8_arr;

  enum Enum { kZero, kOne } enum_one = kOne;
  Enum &enum_ref = enum_one;

  // BREAK(TestSubscript)
}

static void TestArrayDereference() {
  int arr_1d[2] = {1, 2};
  int arr_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};

  // BREAK(TestArrayDereference)
}

// Referenced by TestCStyleCast
namespace ns {

typedef int myint;

class Foo {};

namespace inner {

using mydouble = double;

class Foo {};

} // namespace inner

} // namespace ns

static void TestCStyleCast() {
  int a = 1;
  int *ap = &a;
  void *vp = &a;
  int arr[2] = {1, 2};

  int na = -1;
  float f = 1.1;

  typedef int myint;

  myint myint_ = 1;
  ns::myint ns_myint_ = 2;
  ns::Foo ns_foo_;
  ns::Foo *ns_foo_ptr_ = &ns_foo_;

  ns::inner::mydouble ns_inner_mydouble_ = 1.2;
  ns::inner::Foo ns_inner_foo_;
  ns::inner::Foo *ns_inner_foo_ptr_ = &ns_inner_foo_;

  float finf = std::numeric_limits<float>::infinity();
  float fnan = std::numeric_limits<float>::quiet_NaN();
  float fsnan = std::numeric_limits<float>::signaling_NaN();
  float fmax = std::numeric_limits<float>::max();
  float fdenorm = std::numeric_limits<float>::denorm_min();

  // BREAK(TestCStyleCastBuiltins)
  // BREAK(TestCStyleCastBasicType)
  // BREAK(TestCStyleCastPointer)
  // BREAK(TestCStyleCastNullptrType)

  struct InnerFoo {
    int a;
    int b;
  };

  InnerFoo ifoo;
  (void)ifoo;

  int arr_1d[] = {1, 2, 3, 4};
  int arr_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};

  // BREAK(TestCStyleCastArray)
  // BREAK(TestCStyleCastReference)
}

// Referenced by TestCxxCast
struct CxxVirtualBase {
  int a;
  virtual ~CxxVirtualBase(){};
};
struct CxxVirtualParent : CxxVirtualBase {
  int b;
};

static void TestCxxCast() {
  struct CxxBase {
    int a;
    int b;
  };
  struct CxxParent : CxxBase {
    long long c;
    short d;
  };

  enum UEnum { kUZero, kUOne, kUTwo };
  enum class SEnum { kSZero, kSOne };

  UEnum u_enum = kUTwo;
  SEnum s_enum = SEnum::kSOne;

  typedef int td_int_t;
  typedef int *td_int_ptr_t;
  typedef int &td_int_ref_t;
  typedef SEnum td_senum_t;
  td_int_t td_int = 13;
  td_int_ptr_t td_int_ptr = &td_int;
  td_int_ref_t td_int_ref = td_int;
  td_senum_t td_senum = s_enum;

  CxxParent parent;
  parent.a = 1;
  parent.b = 2;
  parent.c = 3;
  parent.d = 4;

  CxxBase *base = &parent;

  int arr[] = {1, 2, 3, 4, 5};
  int *ptr = arr;

  // BREAK(TestCxxStaticCast)
  // BREAK(TestCxxReinterpretCast)

  CxxVirtualParent v_parent;
  v_parent.a = 1;
  v_parent.b = 2;
  CxxVirtualBase *v_base = &v_parent;

  // BREAK(TestCxxDynamicCast)
}

void TestCastInheritedTypes() {
  struct CxxEmpty {};
  struct CxxA {
    short a;
  };
  struct CxxB {
    long long b;
  };
  struct CxxC : CxxEmpty, CxxA, CxxB {
    int c;
  };
  struct CxxD {
    long long d;
  };
  struct CxxE : CxxD, CxxC {
    int e;
  };

  CxxA a{1};
  CxxB b{2};
  CxxC c;
  c.a = 3;
  c.b = 4;
  c.c = 5;
  CxxD d{6};
  CxxE e;
  e.a = 7;
  e.b = 8;
  e.c = 9;
  e.d = 10;
  e.e = 11;

  struct CxxVC : virtual CxxA, virtual CxxB {
    int c;
  };
  struct CxxVE : CxxD, CxxVC {
    int e;
  };

  CxxVC vc;
  vc.a = 12;
  vc.b = 13;
  vc.c = 14;
  CxxVE ve;
  ve.a = 15;
  ve.b = 16;
  ve.c = 17;
  ve.d = 18;
  ve.e = 19;

  CxxB *e_as_b = &e;
  CxxB *ve_as_b = &ve;

  // BREAK(TestCastBaseToDerived)
  // BREAK(TestCastDerivedToBase)
}

// Referenced by TestQualifiedId.
namespace ns {

int i = 1;

namespace ns {

int i = 2;

} // namespace ns

} // namespace ns

static void TestQualifiedId() {
  // BREAK(TestQualifiedId)
}

namespace outer {

namespace inner {

class Vars {
public:
  inline static double inline_static = 1.5;
  static constexpr int static_constexpr = 2;
  static const unsigned int static_const;

  struct Nested {
    static const int static_const;
  };
};

const unsigned int Vars::static_const = 3;
const int Vars::Nested::static_const = 10;

using MyVars = Vars;

} // namespace inner

class Vars {
public:
  inline static double inline_static = 4.5;
  static constexpr int static_constexpr = 5;
  static const unsigned int static_const;

  struct Nested {
    static const int static_const;
  };
};

const unsigned int Vars::static_const = 6;
const int Vars::Nested::static_const = 20;

} // namespace outer

class Vars {
public:
  inline static double inline_static = 7.5;
  static constexpr int static_constexpr = 8;
  static const unsigned int static_const;

  struct Nested {
    static const int static_const;
  };
};

const unsigned int Vars::static_const = 9;
const int Vars::Nested::static_const = 30;

static void TestStaticConst() {
  Vars vars;
  outer::Vars outer_vars;
  outer::inner::Vars outer_inner_vars;

  using MyVars = Vars;
  using MyOuterVars = outer::Vars;

  MyVars my_vars;
  MyOuterVars my_outer_vars;
  outer::inner::MyVars my_outer_inner_vars;

  // BREAK(TestStaticConstDeclaredInline)
  // BREAK(TestStaticConstDeclaredInlineScoped)
  // BREAK(TestStaticConstDeclaredOutsideTheClass)
  // BREAK(TestStaticConstDeclaredOutsideTheClassScoped)
}

// Referenced by TestTemplateTypes.
template <typename T> struct T_1 {
  static const int cx;
  typedef double myint;

  T_1() {}
  T_1(T x) : x(x) {}
  T x;
};

template <typename T> const int T_1<T>::cx = 42;

template <> const int T_1<int>::cx = 24;

template <typename T1, typename T2> struct T_2 {
  typedef float myint;

  T_2() {}
  T1 x;
  T2 y;
};

namespace ns {

template <typename T> struct T_1 {
  static const int cx;
  typedef int myint;

  T_1() {}
  T_1(T x) : x(x) {}
  T x;
};

template <typename T> const int T_1<T>::cx = 46;

template <> const int T_1<int>::cx = 64;

} // namespace ns

static void TestTemplateTypes() {
  int i;
  int *p = &i;

  { T_1<int> _; }
  { T_1<int *> _; }
  { T_1<int **> _; }
  { T_1<int &> _(i); }
  { T_1<int *&> _(p); }
  { T_1<double> _; }
  { T_2<int, char> _; }
  { T_2<char, int> _; }
  { T_2<T_1<int>, T_1<char>> _; }
  { T_2<T_1<T_1<int>>, T_1<char>> _; }

  { ns::T_1<int> _; }
  { ns::T_1<int *> _; }
  { ns::T_1<int **> _; }
  { ns::T_1<int &> _(i); }
  { ns::T_1<int *&> _(p); }
  { ns::T_1<ns::T_1<int>> _; }
  { ns::T_1<ns::T_1<int *>> _; }
  { ns::T_1<ns::T_1<int **>> _; }
  { ns::T_1<ns::T_1<int &>> _(i); }
  { ns::T_1<ns::T_1<int *&>> _(p); }

  { T_1<int>::myint _ = 0; }
  { T_1<int *>::myint _ = 0; }
  { T_1<int **>::myint _ = 0; }
  { T_1<int &>::myint _ = 0; }
  { T_1<int *&>::myint _ = 0; }
  { T_1<T_1<int>>::myint _ = 0; }
  { T_1<T_1<T_1<int>>>::myint _ = 0; }
  { T_1<T_1<int *>>::myint _ = 0; }
  { T_1<T_1<int **>>::myint _ = 0; }
  { T_1<T_1<int &>>::myint _ = 0; }
  { T_1<T_1<int *&>>::myint _ = 0; }

  { T_2<int, char>::myint _ = 0; }
  { T_2<int *, char &>::myint _ = 0; }
  { T_2<int &, char *>::myint _ = 0; }
  { T_2<T_1<T_1<int>>, T_1<char>>::myint _ = 0; }

  { ns::T_1<int>::myint _ = 0; }
  { ns::T_1<int *>::myint _ = 0; }
  { ns::T_1<int **>::myint _ = 0; }
  { ns::T_1<int &>::myint _ = 0; }
  { ns::T_1<int *&>::myint _ = 0; }
  { ns::T_1<T_1<int>>::myint _ = 0; }
  { ns::T_1<T_1<int *>>::myint _ = 0; }
  { ns::T_1<T_1<int **>>::myint _ = 0; }
  { ns::T_1<T_1<int &>>::myint _ = 0; }
  { ns::T_1<T_1<int *&>>::myint _ = 0; }
  { ns::T_1<ns::T_1<int>>::myint _ = 0; }
  { ns::T_1<ns::T_1<int *>>::myint _ = 0; }
  { ns::T_1<ns::T_1<int **>>::myint _ = 0; }
  { ns::T_1<ns::T_1<int &>>::myint _ = 0; }
  { ns::T_1<ns::T_1<int *&>>::myint _ = 0; }

  (void)T_1<double>::cx;
  (void)ns::T_1<double>::cx;
  (void)ns::T_1<ns::T_1<int>>::cx;

  int T_1 = 2;

  // BREAK(TestTemplateTypes)
  // BREAK(TestTemplateCpp11)
}

template <typename T, typename TAllocator> struct TArray {
  using ElementType = T;
  T t_;
  TAllocator a_;
};

template <int Size> struct Allocator {
  int size = Size;
};

void TestTemplateWithNumericArguments() {
  Allocator<4> a4;
  Allocator<8> a8;
  TArray<int, Allocator<4>> arr;
  decltype(arr)::ElementType *el = 0;

  // BREAK(TestTemplateWithNumericArguments)
}

namespace test_scope {

class Value {
public:
  Value(int x, float y) : x_(x), y_(y) {}

  // Static members
  enum ValueEnum { A, B };
  static double static_var;

private:
  int x_;
  float y_;
};

double Value::static_var = 3.5;

} // namespace test_scope

void TestValueScope() {
  test_scope::Value var(1, 2.5f);
  test_scope::Value &var_ref = var;
  uint64_t z_ = 3;

  // "raw" representation of the Value.
  int bytes[] = {1, 0x40200000};

  auto val_enum = test_scope::Value::A;
  (void)val_enum;
  (void)test_scope::Value::static_var;

  // BREAK(TestValueScope)
  // BREAK(TestReferenceScope)
}

void TestBitField() {
  enum BitFieldEnum : uint32_t { kZero, kOne };

  struct BitFieldStruct {
    uint16_t a : 10;
    uint32_t b : 4;
    bool c : 1;
    bool d : 1;
    int32_t e : 32;
    uint32_t f : 32;
    uint32_t g : 31;
    uint64_t h : 31;
    uint64_t i : 33;
    BitFieldEnum j : 10;
  };

  BitFieldStruct bf;
  bf.a = 0b1111111111;
  bf.b = 0b1001;
  bf.c = 0b0;
  bf.d = 0b1;
  bf.e = 0b1;
  bf.f = 0b1;
  bf.g = 0b1;
  bf.h = 0b1;
  bf.i = 0b1;
  bf.j = BitFieldEnum::kOne;

  struct AlignedBitFieldStruct {
    uint16_t a : 10;
    uint8_t b : 4;
    unsigned char : 0;
    uint16_t c : 2;
  };

  uint32_t data = ~0;
  AlignedBitFieldStruct abf = (AlignedBitFieldStruct &)data;

  // BREAK(TestBitField)
  // BREAK(TestBitFieldScoped)
  // BREAK(TestBitFieldPromotion)
  // BREAK(TestBitFieldWithSideEffects)
}

void TestContextVariables() {
  struct Scope {
    int a = 10;
    const char *ptr = "hello";
  };

  Scope s;

  // BREAK(TestContextVariables)
  // BREAK(TestContextVariablesSubset)
}

// Referenced by TestScopedEnum.
enum class ScopedEnum { kFoo, kBar };
enum class ScopedEnumUInt8 : uint8_t { kFoo, kBar };

void TestScopedEnum() {
  auto enum_foo = ScopedEnum::kFoo;
  auto enum_bar = ScopedEnum::kBar;
  auto enum_neg = (ScopedEnum)-1;

  auto enum_u8_foo = ScopedEnumUInt8::kFoo;
  auto enum_u8_bar = ScopedEnumUInt8::kBar;

  // BREAK(TestScopedEnum)
  // BREAK(TestScopedEnumArithmetic)
  // BREAK(TestScopedEnumWithUnderlyingType)
}

enum UnscopedEnum { kZero, kOne, kTwo };
enum UnscopedEnumUInt8 : uint8_t { kZeroU8, kOneU8, kTwoU8 };
enum UnscopedEnumInt8 : int8_t { kZero8, kOne8, kTwo8 };
enum UnscopedEnumEmpty : uint8_t {};

// UnscopedEnum global_enum = UnscopedEnum::kOne;

void TestUnscopedEnum() {
  auto enum_zero = UnscopedEnum::kZero;
  auto enum_one = UnscopedEnum::kOne;
  auto enum_two = UnscopedEnum::kTwo;

  auto &enum_one_ref = enum_one;
  auto &enum_two_ref = enum_two;

  auto enum_zero_u8 = UnscopedEnumUInt8::kZeroU8;
  auto enum_one_u8 = UnscopedEnumUInt8::kOneU8;
  auto enum_two_u8 = UnscopedEnumUInt8::kTwoU8;

  UnscopedEnumEmpty enum_empty{};

  auto enum_one_8 = UnscopedEnumInt8::kOne8;
  auto enum_neg_8 = (UnscopedEnumInt8)-1;

  // BREAK(TestUnscopedEnum)
  // BREAK(TestUnscopedEnumNegation)
  // BREAK(TestUnscopedEnumWithUnderlyingType)
  // BREAK(TestUnscopedEnumEmpty)
}

void TestTernaryOperator() {
  int i = 1;
  int *pi = &i;
  char c = 2;
  int arr2[2] = {1, 2};
  int arr3[3] = {1, 2, 3};
  double dbl_arr[2] = {1.0, 2.0};
  struct T {
  } t;
  enum EnumA { kOneA = 1, kTwoA } a_enum = kTwoA;
  enum EnumB { kOneB = 1 } b_enum = kOneB;
  // BREAK(TestTernaryOperator)
}

void TestSizeOf() {
  int i = 1;
  int *p = &i;
  int arr[] = {1, 2, 3};

  struct SizeOfFoo {
    int x, y;
  } foo;

  // BREAK(TestSizeOf)
}

void TestBuiltinFunction_Log2() {
  struct Foo {
  } foo;

  enum CEnum { kFoo = 129 } c_enum = kFoo;
  enum class CxxEnum { kFoo = 129 } cxx_enum = CxxEnum::kFoo;

  CEnum &c_enum_ref = c_enum;
  CxxEnum &cxx_enum_ref = cxx_enum;

  // BREAK(TestBuiltinFunction_Log2)
}

void TestBuiltinFunction_findnonnull() {
  uint8_t array_of_uint8[] = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                              0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
  uint8_t *pointer_to_uint8 = array_of_uint8;

  int *array_of_pointers[] = {(int *)1, (int *)1, (int *)0, (int *)0, (int *)1};
  int **pointer_to_pointers = array_of_pointers;

  // BREAK(TestBuiltinFunction_findnonnull)
}

void TestPrefixIncDec() {
  auto enum_foo = ScopedEnum::kFoo;
  int i = 1;

  // BREAK(TestPrefixIncDec)
  // BREAK(TestPostfixIncDec)
}

void TestDereferencedType() {
  struct TTuple {
    int x = 1;
  };
  using TPair = TTuple;

  TPair p{};
  const TPair &p_ref = p;
  const TPair *p_ptr = &p;

  // BREAK(TestDereferencedType)
}

void TestMemberFunctionCall() {
  struct C {
    int m() { return 1; }
  };

  C c;
  c.m();

  // BREAK(TestMemberFunctionCall)
}

void TestCompositeAssignment() {
  int i = 10;
  float f = 1.5f;
  float *p = &f;

  enum Enum { ONE, TWO };
  Enum eOne = ONE;
  Enum eTwo = TWO;

  // BREAK(TestAssignment)
  // BREAK(TestCompositeAssignmentInvalid)
  // BREAK(TestCompositeAssignmentAdd)
  // BREAK(TestCompositeAssignmentSub)
  // BREAK(TestCompositeAssignmentMul)
  // BREAK(TestCompositeAssignmentDiv)
  // BREAK(TestCompositeAssignmentRem)
  // BREAK(TestCompositeAssignmentBitwise)
}

void TestSideEffects() {
  int x = 1;
  int xa[] = {1, 2};
  int *p = &x;

  // BREAK(TestSideEffects)
}

void TestUniquePtr() {
  struct NodeU {
    std::unique_ptr<NodeU> next;
    int value;
  };
  auto ptr_node = std::unique_ptr<NodeU>(new NodeU{nullptr, 2});
  ptr_node = std::unique_ptr<NodeU>(new NodeU{std::move(ptr_node), 1});

  std::unique_ptr<char> ptr_null;
  auto ptr_int = std::make_unique<int>(1);
  auto ptr_float = std::make_unique<float>(1.1f);

  auto deleter = [](void const *data) {
    delete static_cast<int const *>(data);
  };
  std::unique_ptr<void, decltype(deleter)> ptr_void(new int(42), deleter);

  // BREAK(TestUniquePtr)
  // BREAK(TestUniquePtrDeref)
  // BREAK(TestUniquePtrCompare)
}

void TestSharedPtr() {
  struct NodeS {
    std::shared_ptr<NodeS> next;
    int value;
  };
  auto ptr_node = std::shared_ptr<NodeS>(new NodeS{nullptr, 2});
  ptr_node = std::shared_ptr<NodeS>(new NodeS{std::move(ptr_node), 1});

  std::shared_ptr<char> ptr_null;
  auto ptr_int = std::make_shared<int>(1);
  auto ptr_float = std::make_shared<float>(1.1f);

  std::weak_ptr<int> ptr_int_weak = ptr_int;

  std::shared_ptr<void> ptr_void = ptr_int;

  // BREAK(TestSharedPtr)
  // BREAK(TestSharedPtrDeref)
  // BREAK(TestSharedPtrCompare)
}

void TestTypeComparison() {
  int i = 1;
  int const *const icpc = &i;
  int *ip = &i;
  int const *const *const icpcpc = &icpc;
  int **ipp = &ip;

  using MyInt = int;
  using MyPtr = MyInt *;
  MyInt mi = 2;
  MyPtr *mipp = ipp;

  using MyConstInt = const int;
  using MyConstPtr = MyConstInt *const;
  MyConstPtr *const micpcpc = icpcpc;

  char c = 2;
  signed char sc = 65;
  const char cc = 66;
  using mychar = char;
  mychar mc = 67;

  // BREAK(TestTypeComparison)
}

static void TestTypeDeclaration() {
  wchar_t wchar = 0;
  char16_t char16 = 0;
  char32_t char32 = 0;

  using mylong = long;
  mylong my_long = 1;

  // BREAK(TestBasicTypeDeclaration)
  // BREAK(TestUserTypeDeclaration)
}

static void TestTypeVsIdentifier() {
  struct StructOrVar {
    int x = 1;
  } s;
  short StructOrVar = 2;

  class ClassOrVar {
  public:
    int x = 3;
  };
  ClassOrVar ClassOrVar;

  union UnionOrVar {
    int x;
  } u;
  int UnionOrVar[2] = {1, 2};

  enum EnumOrVar { kFoo, kBar };
  EnumOrVar EnumOrVar = kFoo;

  enum class CxxEnumOrVar { kCxxFoo, kCxxBar };
  CxxEnumOrVar CxxEnumOrVar = CxxEnumOrVar::kCxxFoo;

  int OnlyVar = 4;

  // BREAK(TestTypeVsIdentifier)
}

static void TestSeparateParsing() {
  struct StructA {
    int a_;
  } a{1};

  struct StructB {
    int b_;
  } b{2};

  struct StructC : public StructA, public StructB {
    int c_;
  } c{{3}, {4}, 5};

  struct StructD : public StructC {
    int d_;
  } d{{{6}, {7}, 8}, 9};

  // BREAK(TestSeparateParsing)
  // BREAK(TestSeparateParsingWithContextVars)
}

// Used by TestRegistersNoDollar
int rcx = 42;

struct RegisterCtx {
  int rbx = 42;

  void TestRegisters() {
    int rax = 42;

    // BREAK(TestRegisters)
    // BREAK(TestRegistersNoDollar)
  }
};

static void TestCharParsing() {
  // BREAK(TestCharParsing)
}

static void TestStringParsing() {
  // BREAK(TestStringParsing)
}

namespace test_binary {

void main() {
  // BREAK(TestSymbols)

  TestMethods tm;

  TestArithmetic();
  TestBitwiseOperators();
  TestPointerArithmetic();
  TestLogicalOperators();
  TestLocalVariables();
  TestMemberOf();
  TestMemberOfInheritance();
  TestMemberOfAnonymousMember();
  TestGlobalVariableLookup();
  tm.TestInstanceVariables();
  TestIndirection();
  tm.TestAddressOf(42);
  TestSubscript();
  TestCStyleCast();
  TestCxxCast();
  TestCastInheritedTypes();
  TestQualifiedId();
  TestStaticConst();
  TestTypeDeclaration();
  TestTemplateTypes();
  TestTemplateWithNumericArguments();
  TestValueScope();
  TestBitField();
  TestContextVariables();
  TestPrefixIncDec();
  TestScopedEnum();
  TestUnscopedEnum();
  TestTernaryOperator();
  TestSizeOf();
  TestBuiltinFunction_Log2();
  TestBuiltinFunction_findnonnull();
  TestArrayDereference();
  TestDereferencedType();
  TestMemberFunctionCall();
  TestCompositeAssignment();
  TestSideEffects();
  TestUniquePtr();
  TestSharedPtr();
  TestTypeComparison();
  TestTypeVsIdentifier();
  TestSeparateParsing();

  RegisterCtx rc;
  rc.TestRegisters();

  TestCharParsing();
  TestStringParsing();

  // BREAK HERE
}

} // namespace test_binary

int main() { test_binary::main(); }
