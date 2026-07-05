#include <limits>

enum Enum {
  enum_case1 = 1,
  enum_case2 = 2,
};

enum EnumBool : bool {
  enum_bool_case1 = false,
  enum_bool_case2 = true,
};

enum class ScopedEnum {
  scoped_enum_case1 = 1,
  scoped_enum_case2 = 2,
};

enum class ScopedCharEnum : char {
  case1 = 1,
  case2 = 2,
};

enum class ScopedLongLongEnum : long long {
  case0 = std::numeric_limits<long long>::min(),
  case1 = 1,
  case2 = std::numeric_limits<long long>::max(),
};

struct A {
  const static int int_val = 1;
  const static int int_val_with_address = 2;
  inline const static int inline_int_val = 3;
  const static bool bool_val = true;

  const static auto char_max = std::numeric_limits<char>::max();
  const static auto schar_max = std::numeric_limits<signed char>::max();
  const static auto uchar_max = std::numeric_limits<unsigned char>::max();
  const static auto int_max = std::numeric_limits<int>::max();
  const static auto uint_max = std::numeric_limits<unsigned>::max();
  const static auto long_max = std::numeric_limits<long>::max();
  const static auto ulong_max = std::numeric_limits<unsigned long>::max();
  const static auto longlong_max = std::numeric_limits<long long>::max();
  const static auto ulonglong_max =
      std::numeric_limits<unsigned long long>::max();
  const static auto wchar_max = std::numeric_limits<wchar_t>::max();

  const static auto char_min = std::numeric_limits<char>::min();
  const static auto schar_min = std::numeric_limits<signed char>::min();
  const static auto uchar_min = std::numeric_limits<unsigned char>::min();
  const static auto int_min = std::numeric_limits<int>::min();
  const static auto uint_min = std::numeric_limits<unsigned>::min();
  const static auto long_min = std::numeric_limits<long>::min();
  const static auto ulong_min = std::numeric_limits<unsigned long>::min();
  const static auto longlong_min = std::numeric_limits<long long>::min();
  const static auto ulonglong_min =
      std::numeric_limits<unsigned long long>::min();
  const static auto wchar_min = std::numeric_limits<wchar_t>::min();

  const static Enum enum_val = enum_case2;
  const static EnumBool enum_bool_val = enum_bool_case1;
  const static ScopedEnum scoped_enum_val = ScopedEnum::scoped_enum_case2;
  const static ScopedEnum not_enumerator_scoped_enum_val = static_cast<ScopedEnum>(5);
  const static ScopedEnum not_enumerator_scoped_enum_val_2 =
      static_cast<ScopedEnum>(7);
  const static ScopedCharEnum scoped_char_enum_val = ScopedCharEnum::case2;
  const static ScopedLongLongEnum scoped_ll_enum_val_neg =
      ScopedLongLongEnum::case0;
  const static ScopedLongLongEnum scoped_ll_enum_val =
      ScopedLongLongEnum::case2;
};

const int A::int_val_with_address;

struct ClassWithOnlyConstStatic {
  const static int member = 3;
};

struct ClassWithConstexprs {
  constexpr static int member = 2;
  constexpr static Enum enum_val = enum_case2;
  constexpr static ScopedEnum scoped_enum_val = ScopedEnum::scoped_enum_case2;
} cwc;

struct ClassWithEnumAlias {
  using EnumAlias = ScopedEnum;
  static constexpr EnumAlias enum_alias = ScopedEnum::scoped_enum_case2;

  using EnumAliasAlias = EnumAlias;
  static constexpr EnumAliasAlias enum_alias_alias =
      ScopedEnum::scoped_enum_case1;
};

namespace ns {
struct Foo {
  constexpr static int mem = 10;

  void bar() { return; }
};
} // namespace ns

struct Foo {
  constexpr static int mem = -29;
};

int func() {
  Foo f1;
  ns::Foo f2;
  f2.bar();
  return ns::Foo::mem + Foo::mem;
}

int main() {
  A a;

  auto char_max = A::char_max;
  auto schar_max = A::schar_max;
  auto uchar_max = A::uchar_max;
  auto int_max = A::int_max;
  auto uint_max = A::uint_max;
  auto long_max = A::long_max;
  auto ulong_max = A::ulong_max;
  auto longlong_max = A::longlong_max;
  auto ulonglong_max = A::ulonglong_max;
  auto wchar_max = A::wchar_max;

  auto char_min = A::char_min;
  auto schar_min = A::schar_min;
  auto uchar_min = A::uchar_min;
  auto int_min = A::int_min;
  auto uint_min = A::uint_min;
  auto long_min = A::long_min;
  auto ulong_min = A::ulong_min;
  auto longlong_min = A::longlong_min;
  auto ulonglong_min = A::ulonglong_min;
  auto wchar_min = A::wchar_min;

  int member_copy = ClassWithOnlyConstStatic::member;

  Enum e = A::enum_val;
  ScopedEnum se = A::scoped_enum_val;
  se = A::not_enumerator_scoped_enum_val;
  ScopedCharEnum sce = A::scoped_char_enum_val;
  ScopedLongLongEnum sle = A::scoped_ll_enum_val;

  auto enum_alias_val = ClassWithEnumAlias::enum_alias;
  auto enum_alias_alias_val = ClassWithEnumAlias::enum_alias_alias;
  auto ret = func();

  return 0; // break here
}
