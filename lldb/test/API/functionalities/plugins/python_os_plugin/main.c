#include <stdio.h>

int main (int argc, char const *argv[], char const *envp[])
{
    puts("stop here"); // Set breakpoint here
    puts("hello");
    puts("don't stop here on tid-specific breakpoint"); // Set tid-specific breakpoint here
    return 0;
}
