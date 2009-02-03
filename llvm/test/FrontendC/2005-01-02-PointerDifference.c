// RUN: %llvmgcc -xc %s -c -o - | llvm-dis | grep -v div

int Diff(int *P, int *Q) { return P-Q; }
