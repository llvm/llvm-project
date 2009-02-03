/*
 * Check that we can compile helloworld
 * RUN: llvmc %s -o %t
 * RUN: ./%t | grep hello
 */

#include <stdio.h>

int main() {
    printf("hello\n");
    return 0;
}
