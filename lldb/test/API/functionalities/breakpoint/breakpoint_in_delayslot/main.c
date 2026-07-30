#include <stdio.h>

foo (int a, int b)
{
    int c;
    if (a<=b)
        c=b-a;
    else
        c=b+a;
    return c;
}

int main()
{
    int a=7, b=8, c;
    
    c = foo(a, b);

return 0;
}

