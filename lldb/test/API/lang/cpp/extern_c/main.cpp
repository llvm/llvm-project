#include <stdio.h>
#include <stdint.h>

extern "C"
{
   int foo();
};

int foo()
{
    puts("foo");
    return 2;
}

int main (int argc, char const *argv[], char const *envp[])
{          
    foo();
    return 0; //% self.expect("expression -- foo()", substrs = ['2'])
}

