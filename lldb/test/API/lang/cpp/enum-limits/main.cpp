enum class U8Enum : unsigned char {
  Min = 0,
  Max = 255,
};
enum class I8Enum : signed char {
  Min = -128,
  Max = 127,
};
enum class U16Enum : unsigned short {
  Min = 0,
  Max = 65535,
};
enum class I16Enum : short {
  Min = -32768,
  Max = 32767,
};
enum class U32Enum : unsigned {
  Min = 0,
  MaxMinusOne = 4294967294,
  Max = 4294967295,
};
enum class I32Enum : int {
  Min = -2147483648,
  Max = 2147483647,
};
enum class U64Enum : unsigned long long {
  Min = 0,
  MaxMinusOne = 18446744073709551614ULL,
  Max = 18446744073709551615ULL,
};
enum class I64Enum : long long {
  Min = -9223372036854775807LL - 1,
  MinPlusOne = -9223372036854775807LL,
  MaxMinusOne = 9223372036854775806LL,
  Max = 9223372036854775807LL,
};

int main() {
  auto u8 = U8Enum::Max;
  auto i8 = I8Enum::Max;
  auto u16 = U16Enum::Max;
  auto i16 = I16Enum::Max;
  auto u32 = U32Enum::Max;
  auto i32 = I32Enum::Max;
  auto u64 = U64Enum::Max;
  auto i64 = I64Enum::Max;
  return 0;
}
