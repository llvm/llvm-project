// REQUIRES: lld, x86

// A public symbol normally restates an entry that ObjectFilePECOFF already
// added from the PE export table. The restatement must be marked Additional
// rather than becoming a second lookup result, or callers that expect a name
// to resolve to exactly one symbol stop finding it.

// RUN: %build --compiler=clang-cl --arch=64 --nodefaultlib -o %t.exe -- %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 lldb-test symtab %t.exe \
// RUN:   --find-symbols-by-regex="exported_" | FileCheck %s

extern "C" {
__declspec(dllexport) int exported_global = 1;
__declspec(dllexport) int exported_function() { return exported_global; }
}

int main() { return exported_function(); }

// CHECK-DAG: Data{{.*}}exported_global
// CHECK-DAG: Additional{{.*}}exported_global
// CHECK-DAG: Code{{.*}}exported_function
// CHECK-DAG: Additional{{.*}}exported_function
