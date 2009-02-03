// PR947
// RUN: %llvmgcc %s -c -o - 

void foo() {
    void *ptr;
  label:
    ptr = &&label;

    goto *ptr;
  }
