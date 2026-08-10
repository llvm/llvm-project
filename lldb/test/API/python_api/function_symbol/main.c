#include <stdio.h>

// This simple program is to test the lldb Python APIs SBTarget, SBFrame,
// SBFunction, SBSymbol, and SBAddress.
//
// When stopped on breakpoint 1, we can get the line entry using SBFrame API
// SBFrame.GetLineEntry().  We'll get the start address for the line entry
// with the SBAddress type, resolve the symbol context using the SBTarget API
// SBTarget.ResolveSymbolContextForAddress() in order to get the SBSymbol.
//
// We then stop at breakpoint 2, get the SBFrame, and the SBFunction object.
//
// The address from calling GetStartAddress() on the symbol and the function
// should point to the same address, and we also verify that.

int a(int);
int b(int);
int c(int);

int a(int val)
{
    if (val <= 1) // Find the line number for breakpoint 1 here.
        val = b(val);
    else if (val >= 3)
        val = c(val);

    return val; // Find the line number for breakpoint 2 here.
}

int b(int val)
{
    return c(val);
}

int c(int val)
{
    return val + 3;
}

int main (int argc, char const *argv[])
{
    int A1 = a(1);  // a(1) -> b(1) -> c(1)
    printf("a(1) returns %d\n", A1);
    
    int B2 = b(2);  // b(2) -> c(2)
    printf("b(2) returns %d\n", B2);
    
    int A3 = a(3);  // a(3) -> c(3)
    printf("a(3) returns %d\n", A3);
    
    return 0;
}
