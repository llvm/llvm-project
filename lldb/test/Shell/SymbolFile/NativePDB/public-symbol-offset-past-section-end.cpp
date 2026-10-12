// REQUIRES: lld, x86

// A public symbol's offset is not guaranteed to agree with
// the size of the section its segment maps to (this can happen with
// real-world PDBs, e.g. after incremental linking).
//
// RUN: %build --compiler=clang-cl --arch=64 --nodefaultlib -o %t.exe -- %s
// RUN: llvm-pdbutil pdb2yaml --all %t.pdb > %t.yaml
// RUN: %python %S/Inputs/corrupt-public-offset.py %t.yaml main 0x0fffffff
// RUN: llvm-pdbutil yaml2pdb %t.yaml -pdb=%t.pdb
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 lldb-test symtab %t.exe \
// RUN:   --find-symbols-by-regex=".*" | FileCheck %s

int global_one = 1;
int global_two = 2;

int main() { return global_one + global_two; }

// `main`'s offset was rewritten to a value past the end of its section, so
// its size estimate must be left at 0 rather than underflowing.
// CHECK-DAG: Code{{.*}}0x0000000000000000 0x00000000 main
// CHECK-DAG: Data{{.*}}global_one
// CHECK-DAG: Data{{.*}}global_two
