#include <stdio.h>

int main()
{
    int c = 1;

    int (^add)(int, int) = ^int(int a, int b)
    {
        return a + b + c; // Set breakpoint 0 here.
    };

    int (^neg)(int) = ^int(int a)
    {
        return -a;
    };

    printf("%d\n", add(3, 4));
    printf("%d\n", neg(-5)); // Set breakpoint 1 here.

    return 0;
}
