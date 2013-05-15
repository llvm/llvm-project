//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

// This simple program is to test the lldb Python API related to frames.

int a(int, char);
int b(int, char);
int c(int, char);

int a(int val, char ch)
{
    int my_val = val;
    char my_ch = ch;
    printf("a(val=%d, ch='%c')\n", val, ch);
    if (val <= 1)
        return b(val+1, ch+1);
    else if (val >= 3)
        return c(val+1, ch+1);

    return val;
}

int b(int val, char ch)
{
    int my_val = val;
    char my_ch = ch;
    printf("b(val=%d, ch='%c')\n", val, ch);
    return c(val+1, ch+1);
}

int c(int val, char ch)
{
    int my_val = val;
    char my_ch = ch;
    printf("c(val=%d, ch='%c')\n", val, ch);
    return val + 3 + ch;
}

int main (int argc, char const *argv[])
{
    int A1 = a(1, 'A');  // a(1, 'A') -> b(2, 'B') -> c(3, 'C')
    printf("a(1, 'A') returns %d\n", A1);
    
    int B2 = b(2, 'B');  // b(2, 'B') -> c(3, 'C')
    printf("b(2, 'B') returns %d\n", B2);
    
    int A3 = a(3, 'A');  // a(3, 'A') -> c(4, 'B')
    printf("a(3, 'A') returns %d\n", A3);
    
    return 0;
}
