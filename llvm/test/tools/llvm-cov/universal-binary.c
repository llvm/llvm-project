// The coverage reader should be able to handle universal binaries

// CHECK: [[@LINE+1]]| 100|int main
int main(int argc, const char *argv[]) {}

// RUN: llvm-profdata merge %S/Inputs/universal-binary.proftext -o %t.profdata
// RUN: llvm-cov show %S/Inputs/universal-binary -instr-profile %t.profdata -path-equivalence=/tmp,%S %s -arch x86_64 | FileCheck %s
// RUN: llvm-cov export %S/Inputs/universal-binary -instr-profile %t.profdata -arch x86_64 2>&1 | FileCheck %S/Inputs/universal-binary.json

// RUN: llvm-cov report %S/Inputs/universal-binary -arch x86_64 -object %S/Inputs/templateInstantiations.covmapping -arch i386 -instr-profile %t.profdata 2>&1 | FileCheck %s --check-prefix=COMBINED
// COMBINED: showTemplateInstantiations.cpp
// COMBINED-NEXT: universal-binary.c

// RUN: not llvm-cov show %S/Inputs/universal-binary -instr-profile %t.profdata -path-equivalence=/tmp,%S %s -arch i386 2>&1 | FileCheck --check-prefix=WRONG-ARCH %s
// WRONG-ARCH: Failed to load coverage

// RUN: not llvm-cov show %S/Inputs/universal-binary -instr-profile %t.profdata -path-equivalence=/tmp,%S %s -arch definitly_a_made_up_architecture 2>&1 | FileCheck --check-prefix=MADE-UP-ARCH %s
// MADE-UP-ARCH: Unknown architecture: definitly_a_made_up_architecture

// RUN: not llvm-cov show %S/Inputs/universal-binary -instr-profile %t.profdata -path-equivalence=/tmp,%S %s -arch=x86_64 -arch=x86_64 2>&1 | FileCheck --check-prefix=TOO-MANY-ARCH %s
// TOO-MANY-ARCH: Number of architectures doesn't match the number of objects
//
// RUN: not llvm-cov report -instr-profile %t.profdata 2>&1 | FileCheck --check-prefix=MISSING-BINARY %s
// MISSING-BINARY: No filenames specified!
