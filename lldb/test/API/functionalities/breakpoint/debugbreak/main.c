#ifdef _MSC_VER
#include <intrin.h>
#define BREAKPOINT_INTRINSIC()    __debugbreak()
#else
#define BREAKPOINT_INTRINSIC()    __asm__ __volatile__ ("int3")
#endif

int
bar(int const *foo)
{
    int count = 0, i = 0;
    for (; i < 10; ++i)
    {
        count += 1;
        BREAKPOINT_INTRINSIC();
        count += 1;
    }
    return *foo;
}

int
main() { int argc = 0; char **argv = (char **)0;

    int foo = 42;
    bar(&foo);
    return 0;
}


