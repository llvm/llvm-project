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

int32_t global = 10; // Watchpoint variable declaration.

int main(int argc, char** argv) {
    int local = 0;
    printf("&global=%p\n", &global);
    printf("about to write to 'global'...\n"); // Set break point at this line.
                                               // When stopped, watch 'global'.
    global = 20;
    local += argc;
    ++local; // Set 2nd break point for disable_then_enable test case.
    printf("local: %d\n", local);
    printf("global=%d\n", global);
}
