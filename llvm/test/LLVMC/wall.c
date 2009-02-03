/*
 * Check that -Wall works as intended
 * RUN: llvmc -Wall %s -o %t
 * RUN: ./%t | grep hello
 */

#include <stdio.h>

int main() {
    printf("hello\n");
    return 0;
}
