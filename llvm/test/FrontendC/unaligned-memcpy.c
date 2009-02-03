// RUN: %llvmgcc %s -S -emit-llvm -o - | llvm-as | llc

void bork() {
  char Qux[33] = {0};
}
