// The C++ front-end was emitting WAY too many inline functions.  This test
// verifies that it does not emit the body of getchar, because it is not used.
// This corresponds to PR459

// RUN: %llvmgxx %s -S -o - | not grep {^i32 .getchar}

#include <stdio.h>
