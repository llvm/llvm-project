#define DEFINE_UNSIGNED_ENUM(suffix, enum_type)                                \
  enum class enum_##suffix : enum_type{Case1 = 200, Case2, Case3};             \
  enum_##suffix var1_##suffix = enum_##suffix ::Case1;                         \
  enum_##suffix var2_##suffix = enum_##suffix ::Case2;                         \
  enum_##suffix var3_##suffix = enum_##suffix ::Case3;                         \
  enum_##suffix var_below_##suffix = static_cast<enum_##suffix>(199);          \
  enum_##suffix var_above_##suffix = static_cast<enum_##suffix>(203);

#define DEFINE_SIGNED_ENUM(suffix, enum_type)                                  \
  enum class enum_##suffix : enum_type{Case1 = -2, Case2, Case3};              \
  enum_##suffix var1_##suffix = enum_##suffix ::Case1;                         \
  enum_##suffix var2_##suffix = enum_##suffix ::Case2;                         \
  enum_##suffix var3_##suffix = enum_##suffix ::Case3;                         \
  enum_##suffix var_below_##suffix = static_cast<enum_##suffix>(-3);           \
  enum_##suffix var_above_##suffix = static_cast<enum_##suffix>(1);

DEFINE_UNSIGNED_ENUM(uc, unsigned char)
DEFINE_SIGNED_ENUM(c, signed char)
DEFINE_UNSIGNED_ENUM(us, unsigned short int)
DEFINE_SIGNED_ENUM(s, signed short int)
DEFINE_UNSIGNED_ENUM(ui, unsigned int)
DEFINE_SIGNED_ENUM(i, signed int)
DEFINE_UNSIGNED_ENUM(ul, unsigned long)
DEFINE_SIGNED_ENUM(l, signed long)
DEFINE_UNSIGNED_ENUM(ull, unsigned long long)
DEFINE_SIGNED_ENUM(ll, signed long long)

int main(int argc, char const *argv[]) { return 0; }
