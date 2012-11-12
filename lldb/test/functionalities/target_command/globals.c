//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

char my_global_char = 'X';
const char* my_global_str = "abc";
const char **my_global_str_ptr = &my_global_str;
static int my_static_int = 228;

int main (int argc, char const *argv[])
{
    printf("global char: %c\n", my_global_char);
    
    printf("global str: %s\n", my_global_str);

    printf("argc + my_static_int = %d\n", (argc + my_static_int));
    
    return 0;
}
