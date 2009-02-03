// RUN: %llvmgcc %s -S -o - -O | grep {and.*8589934591}
// PR1721

struct s {
  unsigned long long u33: 33;
} a, b;

// This should turn into a real 33-bit add, not a 64-bit add.
_Bool test(void) {
  return a.u33 + b.u33 != 0;
}
