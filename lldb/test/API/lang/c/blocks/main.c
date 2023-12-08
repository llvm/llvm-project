#include <stdio.h>

struct CG {int x; int y;};

int g(int (^callback)(struct CG)) {
   struct CG cg = {.x=1,.y=2};

   int z = callback(cg); // Set breakpoint 2 here.

   return z;
}

int h(struct CG cg){return 42;}

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

    int (^add_struct)(struct CG) = ^int(struct CG cg)
    {
        return cg.x + cg.y;
    };

    g(add_struct);

    return 0;
}
