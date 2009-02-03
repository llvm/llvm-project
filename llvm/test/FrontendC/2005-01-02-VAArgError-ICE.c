// This file is erroneous, but should not cause the compiler to ICE.
// PR481
// RUN: %llvmgcc %s -S -o /dev/null |& not grep {internal compiler error}

#include <stdarg.h>
int flags(int a, int b, ...) {
        va_list         args;
        va_start(args,a);       // not the last named arg
        foo(args);
}
