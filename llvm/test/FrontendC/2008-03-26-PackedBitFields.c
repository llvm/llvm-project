// RUN: %llvmgcc %s -S -o -


struct S1757 { 
  long double c;
  long int __attribute__((packed)) e:28;
} x;
