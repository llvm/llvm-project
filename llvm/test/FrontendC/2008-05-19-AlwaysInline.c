// RUN: %llvmgcc %s -S -fno-unit-at-a-time -emit-llvm -O0 -o - | not grep sabrina
// RUN: %llvmgcc %s -S -funit-at-a-time -emit-llvm -O0 -o - | not grep sabrina

static inline int sabrina (void) __attribute__((always_inline));
static inline int sabrina (void)
{
  return 13;
}
int bar (void)
{
  return sabrina () + 68;
}
