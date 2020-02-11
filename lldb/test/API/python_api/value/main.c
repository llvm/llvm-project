//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stdio.h>
#include <stdint.h>

// This simple program is to test the lldb Python API SBValue.GetChildAtIndex().

int g_my_int = 100;

const char *days_of_week[7] = { "Sunday",
                                "Monday",
                                "Tuesday",
                                "Wednesday",
                                "Thursday",
                                "Friday",
                                "Saturday" };

const char *weekdays[5] = { "Monday",
                            "Tuesday",
                            "Wednesday",
                            "Thursday",
                            "Friday" };

const char **g_table[2] = { days_of_week, weekdays };

typedef int MyInt;

struct MyStruct
{
  int a;
  int b;
};

int main (int argc, char const *argv[])
{
    uint32_t uinthex = 0xE0A35F10;
    int32_t  sinthex = 0xE0A35F10;

    int i;
    MyInt a = 12345;
    struct MyStruct s = { 11, 22 };
    int *my_int_ptr = &g_my_int;
    printf("my_int_ptr points to location %p\n", my_int_ptr);
    const char **str_ptr = days_of_week;
    for (i = 0; i < 7; ++i)
        printf("%s\n", str_ptr[i]); // Break at this line
                                    // and do str_ptr_val.GetChildAtIndex(5, lldb.eNoDynamicValues, True).
    
    return 0;
}
