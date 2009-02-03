// RUN: %llvmgcc %s -S -o - | llvm-as -o /dev/null -f

const double _Complex x[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; 

