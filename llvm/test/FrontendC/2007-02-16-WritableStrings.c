// Test the -fwritable-strings option.

// RUN: %llvmgcc -O3 -S -o - -emit-llvm -fwritable-strings %s | \
// RUN:    grep {internal global}
// RUN: %llvmgcc -O3 -S -o - -emit-llvm %s | grep {internal constant}

char *X = "foo";

