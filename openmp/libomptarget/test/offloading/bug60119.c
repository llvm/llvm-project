// RUN: %clang-generic -fPIC -shared %S/../Inputs/empty.c -o %T/liba.so
// RUN: %clang-generic -fPIC -shared %S/../Inputs/empty.c -o %T/libb.so
// RUN: %clang-generic -rpath %T -L %T -l a -l b %s -o %t
// RUN: %t

int main() {}
