
// nodefaultlib build: cl -Zi sigsegv.cpp /link /nodefaultlib

#ifdef USE_CRT
#include <stdio.h>
#else
int main();
extern "C"
{
    int _fltused;
    void mainCRTStartup() { main(); }
    void printf(const char*, ...) {}
}
#endif

void crash(bool crash_self)
{
    printf("Before...\n");
    if(crash_self)
    {
        printf("Crashing in 3, 2, 1 ...\n");
        *(volatile int*)nullptr = 0;
    }
    printf("After...\n");
}

int foo(int x, float y, const char* msg)
{
    bool flag = x > y;
    if(flag)
        printf("x = %d, y = %f, msg = %s\n", x, y, msg);
    crash(flag);
    return x << 1;
}

int main()
{
    foo(10, 3.14, "testing");
}

