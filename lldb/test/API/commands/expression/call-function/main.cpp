#include <iostream>
#include <string>
#include <cstring>

struct Five
{
    int number;
    const char *name;
};

Five
returnsFive()
{
    Five my_five = {5, "five"};
    return my_five;
}

unsigned int
fib(unsigned int n)
{
    if (n < 2)
        return n;
    else
        return fib(n - 1) + fib(n - 2);
}

int
add(int a, int b)
{
    return a + b;
}

bool
stringCompare(const char *str)
{
    if (strcmp( str, "Hello world" ) == 0)
        return true;
    else
        return false;
}

int main (int argc, char const *argv[])
{
    std::string str = "Hello world";
    std::cout << str << std::endl;
    std::cout << str.c_str() << std::endl;
    Five main_five = returnsFive();
#if 0
    print str
    print str.c_str()
#endif
    return 0; // Please test these expressions while stopped at this line:
}
