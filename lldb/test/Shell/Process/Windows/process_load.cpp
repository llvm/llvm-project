// clang-format off

// REQUIRES: target-windows
// RUN: %build --compiler=clang-cl -o %t.exe -- %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -o "b main" -o "process launch" -o "process load kernel32.dll" | FileCheck %s

int main(int argc, char *argv[]) {
  return 0;
}

// CHECK: "Loading "kernel32.dll"...ok{{.*}}
// CHECK: Image 0 loaded.
