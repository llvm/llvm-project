#include <stdio.h>

#include <chrono>
#include <thread>

long double outermost_return_long_double (long double my_long_double);

int main (int argc, char const *argv[])
{
    lldb_enable_attach();

    char my_string[] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 0};
    double my_double = 1234.5678;
    long double my_long_double = 1234.5678;

    // For simplicity assume that any cmdline argument means wait for attach.
    if (argc > 1)
    {
        volatile int wait_for_attach=1;
        while (wait_for_attach)
            std::this_thread::sleep_for(std::chrono::microseconds(1));
    }

    printf("my_string=%s\n", my_string);
    printf("my_double=%g\n", my_double);
    outermost_return_long_double (my_long_double);
    return 0;
}
