// RUN: %llvmgcc %s -S -o -

# define pck __attribute__((packed))


struct pck E { 
  unsigned long long l, 
    i : 12, 
    j : 23, 
    k : 29; };

struct E e1;

void foo() {
	e1.k = 5;
}
