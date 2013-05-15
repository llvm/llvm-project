#define T unsigned long
#define T_CSTR "unsigned long"

#ifdef __LP64__
#define T_VALUE_1 110011101111
#define T_VALUE_2 220022202222
#define T_VALUE_3 330033303333
#define T_VALUE_4 440044404444
#else
#define T_VALUE_1 110011101
#define T_VALUE_2 220022202
#define T_VALUE_3 330033303
#define T_VALUE_4 440044404
#endif

#define T_PRINTF_FORMAT "%lu"

#include "basic_type.cpp"
