// RUN: %llvmgcc %s -S -o - | llvm-as | opt -std-compile-opts | \
// RUN:    llvm-dis | grep {foo\[12345\]} | count 5

__asm__ ("foo1");
__asm__ ("foo2");
__asm__ ("foo3");
__asm__ ("foo4");
__asm__ ("foo5");
