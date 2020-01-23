#include "foo.h"

struct bar
{
    int a;
    int b;
};

int
main() { int argc = 0; char **argv = (char **)0;

    struct bar b= { 1, 2 };
    
    foo (&b);

    return 0;
}
