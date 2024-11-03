// Test that the file names in the __debug_line_str section can be decoded.

// REQUIRES: system-darwin

// RUN: %clang -target x86_64-apple-darwin %s -c -o %t.o -gdwarf-5
// RUN: llvm-readobj --sections %t.o | FileCheck %s --check-prefix NAMES
// RUN: xcrun %clang -target x86_64-apple-darwin -o %t.exe %t.o
// RUN: %lldb %t.exe -b -o "target modules dump line-table %s" | FileCheck %s

// NAMES: Name: __debug_line_str

int main(int argc, char **argv) {
  // CHECK: dwarf5-macho.c:[[@LINE+1]]
  return 0;
}
