// RUN: %llvmgcc %s -S -o -

struct X { long double b; unsigned char c; double __attribute__((packed)) d; };
struct X x = { 3.0L, 5, 3.0 };


struct S2504 {
  int e:17;
    __attribute__((packed)) unsigned long long int f; 
} ;
int fails;
 extern struct S2504 s2504; 
void check2504va (int z) { 
  struct S2504 arg, *p;
  long long int i = 0; 
  arg.f = i;
}

