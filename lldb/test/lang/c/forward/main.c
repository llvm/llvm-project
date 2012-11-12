#include <stdio.h>
#include "foo.h"

struct bar
{
    int a;
    int b;
};

int
main (int argc, char const *argv[])
{
    struct bar b= { 1, 2 };
    
    foo (&b);

    return 0;
}
