//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>
#include <stdint.h>


int main (int argc, char const *argv[])
{
    typedef int16_t enum_integer_t;
    enum class DayType : enum_integer_t {
        Monday = -3,
        Tuesday,
        Wednesday,
        Thursday,
        Friday,
        Saturday,
        Sunday,
        kNumDays
    };
    enum_integer_t day_value;
    for (day_value = (enum_integer_t)DayType::Monday - 1; day_value <= (enum_integer_t)DayType::kNumDays + 1; ++day_value)
    {
        DayType day = (DayType)day_value;
        printf("day as int is %i\n", (int)day); // Set break point at this line.
    }
    return 0;
}
