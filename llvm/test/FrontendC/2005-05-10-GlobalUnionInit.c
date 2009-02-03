// RUN: %llvmgcc %s -S -o -

union A {                    // { uint }
  union B { double *C; } D;
} E = { { (double*)12312 } };

