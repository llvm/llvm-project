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

int32_t global = 0; // Watchpoint variable declaration.

static void modify(int32_t &var) {
    ++var;
}

int main(int argc, char** argv) {
    int local = 0;
    printf("&global=%p\n", &global);
    printf("about to write to 'global'...\n"); // Set break point at this line.
                                               // When stopped, watch 'global',
                                               // for the condition "global == 5".
    for (int i = 0; i < 10; ++i)
        modify(global);

    printf("global=%d\n", global);
}
