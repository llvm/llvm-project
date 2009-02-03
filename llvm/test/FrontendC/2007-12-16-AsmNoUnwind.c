// RUN: %llvmgcc %s -S -o - | grep nounwind

void bar() { asm (""); }
