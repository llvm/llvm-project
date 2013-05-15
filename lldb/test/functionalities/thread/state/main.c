//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This test is intended to verify that thread states are properly maintained
// when transitional actions are performed in the debugger.  Most of the logic
// is in the test script.  This program merely provides places where the test
// can create the intended states.

#include <unistd.h>

volatile int g_test = 0;

int addSomething(int a)
{
    return a + g_test;
}

int doNothing()
{
    int temp = 0;   // Set first breakpoint here

    while (!g_test && temp < 5)
    {
        ++temp;
        sleep(1);
    }

    return temp;    // Set second breakpoint here
}

int main ()
{
    int result = doNothing();

    int i = addSomething(result);

    return 0;
}
