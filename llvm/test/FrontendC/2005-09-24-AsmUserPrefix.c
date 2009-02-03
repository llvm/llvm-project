// RUN: %llvmgcc %s -S -o - | llvm-as | opt -std-compile-opts | llc | \
// RUN:    not grep _foo2

void foo() __asm__("foo2");

void bar() {
  foo();
}
