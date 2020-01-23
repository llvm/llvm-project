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
#ifdef SIGNED_ENUM_CLASS_TYPE
    typedef SIGNED_ENUM_CLASS_TYPE enum_integer_t;
    enum class DayType : enum_integer_t {
        Monday = -3,
        Tuesday,
        Wednesday,
        Thursday,
        Friday,
        Saturday,
        Sunday,
        kNumDays
    };
    enum_integer_t day_value;
#else
    typedef UNSIGNED_ENUM_CLASS_TYPE enum_integer_t;
    enum class DayType : enum_integer_t {
        Monday = 200,
        Tuesday,
        Wednesday,
        Thursday,
        Friday,
        Saturday,
        Sunday,
        kNumDays
    };
    enum_integer_t day_value;
#endif

int main() { int argc = 0; char **argv = (char **)0; return 0; }
