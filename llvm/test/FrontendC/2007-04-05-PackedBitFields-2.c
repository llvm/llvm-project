// RUN: %llvmgcc %s -S -o -

# define pck __attribute__((packed))


struct pck F { 
  unsigned long long i : 12, 
    j : 23, 
    k : 27, 
    l; 
}; 
struct F f1;

void foo() {
	f1.l = 5;
}
