// Checks for reading various formats.

// CHECK: [[@LINE+1]]| 100|int main
int main(int argc, const char *argv[]) {}

// RUN: llvm-profdata merge %S/Inputs/binary-formats.proftext -o %t.profdata
// RUN: llvm-cov show %S/Inputs/binary-formats.macho32l -instr-profile %t.profdata -path-equivalence=/tmp,%S %s | FileCheck %s
// RUN: llvm-cov show %S/Inputs/binary-formats.macho64l -instr-profile %t.profdata -path-equivalence=/tmp,%S %s | FileCheck %s
// RUN: llvm-cov show %S/Inputs/binary-formats.macho32b -instr-profile %t.profdata -path-equivalence=/tmp,%S %s | FileCheck %s
// RUN: llvm-cov show %S/Inputs/binary-formats.v3.macho64l -instr-profile %t.profdata -path-equivalence=/tmp,%S %s | FileCheck %s
// RUN: llvm-cov show %S/Inputs/binary-formats.v6.linux64l -instr-profile %t.profdata -path-equivalence=/tmp,%S %s | FileCheck %s

// RUN: llvm-profdata merge %S/Inputs/binary-formats.wasm.proftext -o %t.wasm.profdata
// NOTE: The wasm binary is built with the following command:
//   clang -target wasm32-unknown-wasi %s -o %S/Inputs/binary-formats.v6.wasm32 \
//     -mllvm -enable-name-compression=false \
//     -fprofile-instr-generate -fcoverage-mapping -lwasi-emulated-getpid -lwasi-emulated-mman
// RUN: llvm-cov show %S/Inputs/binary-formats.v6.wasm32 -instr-profile %t.wasm.profdata -path-equivalence=/tmp,%S %s | FileCheck %s

// RUN: llvm-cov export %S/Inputs/binary-formats.macho64l -instr-profile %t.profdata | FileCheck %S/Inputs/binary-formats.canonical.json
