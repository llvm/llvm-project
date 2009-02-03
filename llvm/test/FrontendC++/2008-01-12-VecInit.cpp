// RUN: %llvmgcc -xc++ %s -S -o -
// rdar://5685492

typedef int __attribute__((vector_size(16))) v;
v vt = {1, 2, 3, 4};

