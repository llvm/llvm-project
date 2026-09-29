// RUN: llvm-profdata merge %S/Inputs/compilation_dir.proftext -o %t.profdata
// RUN: llvm-cov show %S/Inputs/compilation_dir.covmapping -instr-profile=%t.profdata -compilation-dir=%S | FileCheck %s
int main() {} // CHECK: [[@LINE]]|      1|int main() {}
