#include <cstdlib>
#include <string>
#include <fstream>
#include <iostream>


#define INLINE inline __attribute__((always_inline))

INLINE int
product (int x, int y)
{
    int result = x * y;
    return result;
}

INLINE int
sum (int a, int b)
{
    int result = a + b;
    return result;
}

int
strange_max (int m, int n)
{
    if (m > n)
        return m;
    else if (n > m)
        return n;
    else
        return 0;
}

int
foo (int i, int j)
{
    if (strange_max (i, j) == i)
        return product (i, j);
    else if (strange_max  (i, j) == j)
        return sum (i, j);
    else
        return product (sum (i, i), sum (j, j));
}

int
main(int argc, char const *argv[])
{

    int array[3];

    array[0] = foo (1238, 78392);
    array[1] = foo (379265, 23674);
    array[2] = foo (872934, 234);

    return 0;
}
