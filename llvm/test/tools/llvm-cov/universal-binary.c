// The coverage reader should be able to handle universal binaries

// CHECK: [[@LINE+1]]| 100|int main
int main(int argc, const char *argv[]) {}

// RUN: llvm-profdata merge %S/Inputs/universal-binary.proftext -o %t.profdata
// RUN: llvm-cov show %S/Inputs/universal-binary -instr-profile %t.profdata -filename-equivalence %s -arch x86_64 | FileCheck %s
// RUN: llvm-cov export %S/Inputs/universal-binary -instr-profile %t.profdata -arch x86_64 2>&1 | FileCheck %S/Inputs/universal-binary.json


// RUN: not llvm-cov show %S/Inputs/universal-binary -instr-profile %t.profdata -filename-equivalence %s -arch i386 2>&1 | FileCheck --check-prefix=WRONG-ARCH %s
// WRONG-ARCH: Failed to load coverage

// RUN: not llvm-cov report -instr-profile %t.profdata 2>&1 | FileCheck --check-prefix=MISSING-BINARY %s
// MISSING-BINARY: No filenames specified!
