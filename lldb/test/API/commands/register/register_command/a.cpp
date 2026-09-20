#include <stdio.h>

long double
return_long_double (long double value)
{
#if defined (__i386__) || defined (__x86_64__)
    float a=2, b=4,c=8, d=16, e=32, f=64, k=128, l=256, add=0;
    __asm__ (
        "int3 ;"
        "flds %1 ;"
        "flds %2 ;"
        "flds %3 ;"
        "flds %4 ;"
        "flds %5 ;"
        "flds %6 ;"
        "flds %7 ;"
        "faddp ;" : "=g" (add) : "g" (a), "g" (b), "g" (c), "g" (d), "g" (e), "g" (f), "g" (k), "g" (l) );  // Set break point at this line.
#endif    // #if defined (__i386__) || defined (__x86_64__)
    return value;
}

long double
outer_return_long_double (long double value)
{
    long double val = return_long_double(value);
    val *= 2 ;
    return val;
}

long double
outermost_return_long_double (long double value)
{
    long double val = outer_return_long_double(value);
    val *= 2 ;
    return val;
}
