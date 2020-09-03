#include <stdio.h>

// This simple program is to demonstrate the capability of the lldb command
// "breakpoint modify -i <count> breakpt-id" to set the number of times a
// breakpoint is skipped before stopping.  Ignore count can also be set upon
// breakpoint creation by 'breakpoint set ... -i <count>'.

int a(int);
int b(int);
int c(int);

int a(int val)
{
    if (val <= 1)
        return b(val);
    else if (val >= 3)
        return c(val); // a(3) -> c(3) Find the call site of c(3).

    return val;
}

int b(int val)
{
    return c(val);
}

int c(int val)
{
    return val + 3; // Find the line number of function "c" here.
}

int main (int argc, char const *argv[])
{
    int A1 = a(1);  // a(1) -> b(1) -> c(1)
    printf("a(1) returns %d\n", A1);
    
    int B2 = b(2);  // b(2) -> c(2) Find the call site of b(2).
    printf("b(2) returns %d\n", B2);
    
    int A3 = a(3);  // a(3) -> c(3) Find the call site of a(3).
    printf("a(3) returns %d\n", A3);
    
    int C1 = c(5); // Find the call site of c in main.
    printf ("c(5) returns %d\n", C1);
    return 0;
}
