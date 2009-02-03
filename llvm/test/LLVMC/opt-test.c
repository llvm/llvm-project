/*
 * Check that the -opt switch works.
 * RUN: llvmc %s -opt -o %t
 * RUN: ./%t | grep hello
 */

#include <stdio.h>

int main() {
    printf("hello\n");
    return 0;
}
