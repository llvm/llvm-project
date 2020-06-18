#include <stdio.h>

void set_char(void)
{
    char i = 'a';
    printf("before (char) i = %c\n", i);
    printf("after  (char) i = %c\n", i);    // Set break point #1. //// break $source:$line
}

void set_uchar(void)
{
    unsigned char i = 'a';
    printf("before (unsigned char) i = %c\n", i);
    printf("after  (unsigned char) i = %c\n", i);   //// break $source:$line
}

void set_short(void)
{
    short i = 33;
    printf("before (short) i = %i\n", i);
    printf("after  (short) i = %i\n", i);   //// break $source:$line
}

void set_ushort(void)
{
    unsigned short i = 33;
    printf("before (unsigned short) i = %i\n", i);
    printf("after  (unsigned short) i = %i\n", i);  // Set break point #2. //// break $source:$line
}

void set_int(void)
{
    int i = 33;
    printf("before (int) i = %i\n", i);
    printf("after  (int) i = %i\n", i); //// break $source:$line
}

void set_uint(void)
{
    unsigned int i = 33;
    printf("before (unsigned int) i = %u\n", i);
    printf("after  (unsigned int) i = %u\n", i);    //// break $source:$line
}

void set_long(void)
{
    long i = 33;
    printf("before (long) i = %li\n", i);
    printf("after  (long) i = %li\n", i);   // Set break point #3. //// break $source:$line
}

void set_ulong(void)
{
    unsigned long i = 33;
    printf("before (unsigned long) i = %lu\n", i);
    printf("after  (unsigned long) i = %lu\n", i);  //// break $source:$line
}

void set_float(void)
{
    float i = 2.25;
    printf("before (float) i = %g\n", i);
    printf("after  (float) i = %g\n", i);   //// break $source:$line
}

void set_double(void)
{
    double i = 2.25;
    printf("before (double) i = %g\n", i);
    printf("after  (double) i = %g\n", i);  // Set break point #4. //// break $source:$line
}

void set_long_double(void)
{
    long double i = 2.25;
    printf("before (long double) i = %Lg\n", i);
    printf("after  (long double) i = %Lg\n", i);    // Set break point #5. //// break $source:$line
}

void set_point (void)
{
    struct point_tag {
        int x;
        int y;
    };
    struct point_tag points_2[2] = {
        {1,2},
        {3,4}
    };
}

int main (int argc, char const *argv[])
{
    // Continue to the breakpoint in set_char()
    set_char();         //// continue; var i; val -set 99 1
    set_uchar();        //// continue; var i; val -set 99 2
    set_short();        //// continue; var i; val -set -42 3
    set_ushort();       //// continue; var i; val -set 42 4
    set_int();          //// continue; var i; val -set -42 5
    set_uint();         //// continue; var i; val -set 42 6
    set_long();         //// continue; var i; val -set -42 7
    set_ulong();        //// continue; var i; val -set 42 8
    set_float();        //// continue; var i; val -set 123.456 9
    set_double();       //// continue; var i; val -set 123.456 10
    set_long_double();  //// continue; var i; val -set 123.456 11
    set_point ();       //// continue
    return 0;
}
