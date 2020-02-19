// A sample program for getting minidumps on Windows.

#include <iostream>

bool
fizz(int x)
{
    return x % 3 == 0;
}

bool
buzz(int x)
{
    return x % 5 == 0;
}

int
main()
{
    int *buggy = 0;

    for (int i = 1; i <= 100; ++i)
    {
        if (fizz(i)) std::cout << "fizz";
        if (buzz(i)) std::cout << "buzz";
        if (!fizz(i) && !buzz(i)) std::cout << i;
        std::cout << '\n';
    }

    return *buggy;
}
