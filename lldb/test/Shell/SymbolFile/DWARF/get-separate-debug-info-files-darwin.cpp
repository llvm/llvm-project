// REQUIRES: system-darwin

// Test SBModule::GetSeparateDebugInfoFiles() for Darwin scenarios:
//   1. DWARF in .o files (no dSYM) -- returns .o file paths
//   2. dSYM present -- returns empty (debug info is self-contained)

struct A {
  int x = 47;
};
A a;
int main() {}

// ============================================================================
// TEST 1: DWARF in .o files (no dSYM) -- returns .o paths
// ============================================================================
// Compile to .o with debug info, then link without generating a dSYM.
// RUN: %clang_host -g -c %s -o %t.main.o
// RUN: %clang_host %t.main.o -o %t.oso -Wl,-no_uuid
// RUN: %lldb -b \
// RUN:   -o "script m = lldb.target.modules[0]; files = m.GetSeparateDebugInfoFiles(); print('OSO_COUNT=' + str(files.GetSize())); [print('OSO_FILE=' + files.GetSpecAtIndex(i).GetFileSpec().fullpath) for i in range(files.GetSize())]" \
// RUN:   %t.oso 2>&1 | FileCheck %s --check-prefix=OSO
//
// With DWARF in .o files, should list the .o file.
// OSO: OSO_COUNT=1
// OSO: OSO_FILE={{.*}}.main.o

// ============================================================================
// TEST 2: dSYM present -- returns empty list
// ============================================================================
// Build with debug info and generate a dSYM bundle.
// RUN: %clang_host -g %s -o %t.dsym_exe
// RUN: dsymutil %t.dsym_exe
// RUN: %lldb -b \
// RUN:   -o "script m = lldb.target.modules[0]; files = m.GetSeparateDebugInfoFiles(); print('DSYM_COUNT=' + str(files.GetSize()))" \
// RUN:   %t.dsym_exe 2>&1 | FileCheck %s --check-prefix=DSYM
//
// With a dSYM, debug info is self-contained -- no separate files.
// DSYM: DSYM_COUNT=0
