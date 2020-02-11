//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct referent {
    const char *p;
};

int main (int argc, char const *argv[])
{
    const char *my_ptr = strdup("hello");
    struct referent *r = malloc(sizeof(struct referent));
    r->p = my_ptr;

    printf("%p\n", r); // break here

    return 0;
}
