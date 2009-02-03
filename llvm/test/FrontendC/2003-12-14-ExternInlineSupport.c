// RUN: %llvmgcc -xc %s -c -o - | llvm-dis | not grep dead_function

extern __inline__ void dead_function() {}
