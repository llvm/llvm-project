#include <stdio.h>

int main (int argc, char const *argv[])
{
    printf ("Hello world!\n");
    puts ("hello");
    // Please test many expressions while stopped at this line:
#if 0
    expression 'a'        // make sure character constant makes it down (this is broken: <rdar://problem/8686536>)
    expression 2          // Test int
    expression 2ull       // Test unsigned long long
    expression 2.234f     // Test float constants
    expression 2.234      // Test double constants
    expression 2+3
    expression argc
    expression argc + 22
    expression argv
    expression argv[0]
    expression argv[1]
    expression argv[-1]
    expression puts("bonjour")                        // Test constant strings...
    expression printf("\t\x68\n")       // Test constant strings that contain the \xXX (TAB, 'h', '\n' should be printed) (this is broken: <rdar://problem/8686536>)
    expression printf("\"\n")       // Test constant strings that contains an escaped double quote char (this is broken: <rdar://problem/8686536>)
    expression printf("\'\n")       // Test constant strings that contains an escaped single quote char (this is broken: <rdar://problem/8686536>)
    expression printf ("one: %i\n", 1)
    expression printf ("1.234 as float: %f\n", 1.234f)
    expression printf ("1.234 as double: %g\n", 1.234)
    expression printf ("one: %i, two: %llu\n", 1, 2ull)
    expression printf ("two: %llu, one: %i\n", 2ull, 1)
    expression random() % 255l
#endif
    return 0;
}
