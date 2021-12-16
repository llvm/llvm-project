#include <stdio.h>

extern int a(int);
extern int b(int);
extern int c(int);
int main (int argc, char const *argv[])
{
    printf ("a(1) returns %d\n", a(1));
    printf ("b(2) returns %d\n", b(2));
    printf ("c(2) returns %d\n", c(2));
}
