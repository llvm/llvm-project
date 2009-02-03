// RUN: %llvmgcc %s -S -o -
struct W {};
struct Y {
  struct W w;
  int i:1;
} __attribute__ ((packed)) y;
