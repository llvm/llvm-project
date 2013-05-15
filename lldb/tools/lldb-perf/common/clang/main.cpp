#include <stdio.h>
#include <stdint.h>
#include <vector>

namespace {
    struct Foo
    {
        int i; int j;
    };
    void doit (const Foo &foo)
    {
        printf ("doit(%i)\n", foo.i);
    }
}
int main (int argc, char const *argv[], char const *envp[])
{
    std::vector<int> ints;
    for (int i=0;i<10;++i)
        ints.push_back(i);
    printf ("hello world\n");
    Foo foo = { 12, 13 };
    doit (foo);
    return 0;
}
