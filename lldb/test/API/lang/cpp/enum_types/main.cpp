//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stdio.h>
#include <stdint.h>


int main (int argc, char const *argv[])
{
#ifdef SIGNED_ENUM_CLASS_TYPE
    typedef SIGNED_ENUM_CLASS_TYPE enum_integer_t;
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
#else
    typedef UNSIGNED_ENUM_CLASS_TYPE enum_integer_t;
    enum class DayType : enum_integer_t {
        Monday = 200,
        Tuesday,
        Wednesday,
        Thursday,
        Friday,
        Saturday,
        Sunday,
        kNumDays
    };
    enum_integer_t day_value;
#endif

    for (day_value = (enum_integer_t)DayType::Monday - 1; day_value <= (enum_integer_t)DayType::kNumDays + 1; ++day_value)
    {
        DayType day = (DayType)day_value;
        printf("day as int is %i\n", (int)day); // Set break point at this line.
    }
    return 0; // Break here for char tests
}
