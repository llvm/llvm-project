// Checks for reading various formats.

// CHECK: [[@LINE+1]]| 100|int main
int main(int argc, const char *argv[]) {}

// RUN: llvm-profdata merge %S/Inputs/binary-formats.proftext -o %t.profdata
// RUN: llvm-cov show %S/Inputs/binary-formats.macho32l -instr-profile %t.profdata -path-equivalence=/tmp,%S %s | FileCheck %s
// RUN: llvm-cov show %S/Inputs/binary-formats.macho64l -instr-profile %t.profdata -path-equivalence=/tmp,%S %s | FileCheck %s
// RUN: llvm-cov show %S/Inputs/binary-formats.macho32b -instr-profile %t.profdata -path-equivalence=/tmp,%S %s | FileCheck %s

// RUN: llvm-cov export %S/Inputs/binary-formats.macho32l -instr-profile %t.profdata | FileCheck %S/Inputs/binary-formats.canonical.json
// RUN: llvm-cov export %S/Inputs/binary-formats.macho64l -instr-profile %t.profdata | FileCheck %S/Inputs/binary-formats.canonical.json
// RUN: llvm-cov export %S/Inputs/binary-formats.macho32b -instr-profile %t.profdata | FileCheck %S/Inputs/binary-formats.canonical.json
