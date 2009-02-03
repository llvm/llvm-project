// RUN: %llvmgcc %s -S -emit-llvm -o - | grep llvm.atomic
// XFAIL: sparc-sun-solaris2|arm|ia64
// Feature currently implemented only for x86, alpha, powerpc.

int* foo(int** a, int* b, int* c) {
return __sync_val_compare_and_swap (a, b, c);
}

int foo2(int** a, int* b, int* c) {
return __sync_bool_compare_and_swap (a, b, c);
}

int* foo3(int** a, int b) {
  return __sync_fetch_and_add (a, b);
}

int* foo4(int** a, int b) {
  return __sync_fetch_and_sub (a, b);
}

int* foo5(int** a, int* b) {
  return __sync_lock_test_and_set (a, b);
}

int* foo6(int** a, int*** b) {
  return __sync_lock_test_and_set (a, b);
}
