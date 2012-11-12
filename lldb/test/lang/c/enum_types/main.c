//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

int main (int argc, char const *argv[])
{
    enum days {
        Monday = 10,
        Tuesday,
        Wednesday,
        Thursday,
        Friday,
        Saturday,
        Sunday,
        kNumDays
    };
    enum days day;
    for (day = Monday - 1; day <= kNumDays + 1; day++)
    {
        printf("day as int is %i\n", (int)day); // Set break point at this line.
    }
    return 0;
}
