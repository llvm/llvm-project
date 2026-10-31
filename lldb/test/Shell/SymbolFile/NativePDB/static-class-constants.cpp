// REQUIRES: lld, x86

// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -GS- -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: %lldb -f %t.exe -o "type lookup Foo" | FileCheck %s

enum class U8Enum : unsigned char {
  Max = 255,
};
enum class I8Enum : char {
  Min = 127,
  Max = -128,
};
enum class U16Enum : unsigned short {
  Max = 65535,
};
enum class I16Enum : short {
  Min = -32768,
  Max = 32767,
};
enum class U32Enum : unsigned {
  Max = 4294967295U,
};
enum class I32Enum : int {
  Min = -2147483648,
  Max = 2147483647,
};
enum class U64Enum : unsigned long long {
  Max = 18446744073709551615ULL,
};
enum class I64Enum : long long {
  Min = -9223372036854775807LL - 1,
  Max = 9223372036854775807LL,
};

// CHECK: struct Foo {
struct Foo {
  // CHECK-NEXT: static const U8Enum u8_enum_max = 255Ui8;
  static constexpr U8Enum u8_enum_max = U8Enum::Max;

  // CHECK-NEXT: static const I8Enum i8_enum_min = 127i8;
  static constexpr I8Enum i8_enum_min = I8Enum::Min;
  // CHECK-NEXT: static const I8Enum i8_enum_max = -128i8;
  static constexpr I8Enum i8_enum_max = I8Enum::Max;

  // CHECK-NEXT: static const U16Enum u16_enum_max = 65535Ui16;
  static constexpr U16Enum u16_enum_max = U16Enum::Max;

  // CHECK-NEXT: static const I16Enum i16_enum_min = -32768i16;
  static constexpr I16Enum i16_enum_min = I16Enum::Min;
  // CHECK-NEXT: static const I16Enum i16_enum_max = 32767i16;
  static constexpr I16Enum i16_enum_max = I16Enum::Max;

  // CHECK-NEXT: static const U32Enum u32_enum_max = 4294967295U;
  static constexpr U32Enum u32_enum_max = U32Enum::Max;

  // CHECK-NEXT: static const I32Enum i32_enum_min = -2147483648;
  static constexpr I32Enum i32_enum_min = I32Enum::Min;
  // CHECK-NEXT: static const I32Enum i32_enum_max = 2147483647;
  static constexpr I32Enum i32_enum_max = I32Enum::Max;

  // CHECK-NEXT: static const U64Enum u64_enum_max = 18446744073709551615ULL;
  static constexpr U64Enum u64_enum_max = U64Enum::Max;

  // CHECK-NEXT: static const I64Enum i64_enum_min = -9223372036854775808LL;
  static constexpr I64Enum i64_enum_min = I64Enum::Min;
  // CHECK-NEXT: static const I64Enum i64_enum_max = 9223372036854775807LL;
  static constexpr I64Enum i64_enum_max = I64Enum::Max;

  // CHECK-NEXT: static const unsigned char u8_max = 255Ui8;
  static constexpr unsigned char u8_max = 255;

  // CHECK-NEXT: static const char i8_min = -128i8;
  static constexpr char i8_min = -128;
  // CHECK-NEXT: static const char i8_max = 127i8;
  static constexpr char i8_max = 127;

  // CHECK-NEXT: static const unsigned short u16_max = 65535Ui16;
  static constexpr unsigned short u16_max = 65535;

  // CHECK-NEXT: static const short i16_min = -32768i16;
  static constexpr short i16_min = -32767 - 1;
  // CHECK-NEXT: static const short i16_max = 32767i16;
  static constexpr short i16_max = 32767;

  // CHECK-NEXT: static const unsigned int u32_max = 4294967295U;
  static constexpr unsigned u32_max = 4294967295;

  // CHECK-NEXT: static const int i32_min = -2147483648;
  static constexpr int i32_min = -2147483648;
  // CHECK-NEXT: static const int i32_max = 2147483647;
  static constexpr int i32_max = 2147483647;

  // CHECK-NEXT: static const unsigned long long u64_max = 18446744073709551615ULL;
  static constexpr unsigned long long u64_max = 18446744073709551615ULL;

  // CHECK-NEXT: static const long long i64_min = -9223372036854775808LL;
  static constexpr long long i64_min = -9223372036854775807LL - 1;
  // CHECK-NEXT: static const long long i64_max = 9223372036854775807LL;
  static constexpr long long i64_max = 9223372036854775807LL;

  // CHECK-NEXT: int i;
  int i;
};

int main() {
  Foo f{42};
  return f.i;
}
