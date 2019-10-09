// clang-format off
// REQUIRES: lld

// Test that we can set display source of functions.
// RUN: %build --compiler=clang-cl --nodefaultlib -o %t.exe -- %s 
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/source-list.lldbinit | FileCheck %s



// Some context lines before
// the function.


int main(int argc, char **argv) {
  // Here are some comments.
  // That we should print when listing source.
  return 0;
}

// Some context lines after
// the function.

// check lines go at the end so that line numbers stay stable when
// changing this file.

// CHECK: (lldb) source list -n main
// CHECK: File: {{.*}}source-list.cpp
// CHECK:    10
// CHECK:    11    // Some context lines before
// CHECK:    12   // the function.
// CHECK:    13
// CHECK:    14
// CHECK:    15   int main(int argc, char **argv) {
// CHECK:    16     // Here are some comments.
// CHECK:    17     // That we should print when listing source.
// CHECK:    18     return 0;
// CHECK:    19   }
// CHECK:    20
// CHECK:    21   // Some context lines after
// CHECK:    22   // the function.
// CHECK:    23
