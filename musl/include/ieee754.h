#ifndef __IEEE754_H__
#define __IEEE754_H__
#include <endian.h>

#if __BYTE_ORDER != __LITTLE_ENDIAN
#error This hack only fits Little-Endian architectures
#endif

#define IEEE754_FLOAT_BIAS 0x7f
#define IEEE754_DOUBLE_BIAS 0x3ff
#define IEEE854_LONG_DOUBLE_BIAS 0x3fff

union ieee754_float
{
  struct
  {
    unsigned int mantissa:23;
    unsigned int exponent:8;
    unsigned int negative:1;
  } ieee;
};

union ieee754_double
{
  struct
  {
    unsigned int mantissa1:32;
    unsigned int mantissa0:20;
    unsigned int exponent:11;
    unsigned int negative:1;
  } ieee;
};

union ieee854_long_double
{
  struct
  {
    unsigned int mantissa1:32;
    unsigned int mantissa0:32;
    unsigned int exponent:15;
    unsigned int negative:1;
    unsigned int empty:16;
  } ieee;
};

#endif
