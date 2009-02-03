// RUN: %llvmgcc %s -S -o -

#ifdef __APPLE__
/* test that X is layed out correctly when this pragma is used. */
#pragma options align=mac68k
#endif

struct S {
  unsigned A;
  unsigned short B;
} X;

