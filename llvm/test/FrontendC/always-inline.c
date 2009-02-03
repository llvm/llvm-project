// RUN: %llvmgcc -S %s -o - | grep call | not grep foo

void bar() {
}

inline void __attribute__((__always_inline__)) foo() {
  bar();
}

void i_want_bar() {
  foo();
}
