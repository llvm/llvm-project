#ifndef OTHER_H
#define OTHER_H

__attribute__((noinline)) void not_inlined_fn() {};

__attribute__((always_inline)) inline void inlined_fn() { not_inlined_fn(); }
#endif // OTHER_H