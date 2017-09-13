// RUN: llvm-profdata merge %S/Inputs/path_equivalence.proftext -o %t.profdata
// RUN: llvm-cov show %S/Inputs/path_equivalence.covmapping -instr-profile=%t.profdata -path-equivalence=/tmp,%S | FileCheck %s

int main() {} // CHECK: [[@LINE]]|      1|int main() {}
