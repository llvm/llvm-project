// REQUIRES: system-linux

// Test SBModule::GetSeparateDebugInfoFiles() for three scenarios:
//   1. No split DWARF -- returns empty list
//   2. Split DWARF with .dwo files -- returns DWO file paths
//   3. Split DWARF with .dwp file -- returns DWP file path

struct A {
  int x = 47;
};
A a;
int main() {}

// ============================================================================
// TEST 1: No split DWARF -- GetSeparateDebugInfoFiles returns empty
// ============================================================================
// RUN: %clang_host -g -c %s -o %t.nosplit.o
// RUN: %clang_host %t.nosplit.o -o %t.nosplit
// RUN: %lldb -b \
// RUN:   -o "script m = lldb.target.modules[0]; files = m.GetSeparateDebugInfoFiles(); print('NOSPLIT_COUNT=' + str(files.GetSize()))" \
// RUN:   %t.nosplit 2>&1 | FileCheck %s --check-prefix=NOSPLIT
//
// NOSPLIT: NOSPLIT_COUNT=0

// ============================================================================
// TEST 2: Split DWARF with .dwo files -- returns DWO paths
// ============================================================================
// RUN: %clang_host -gsplit-dwarf -gdwarf-5 -c %s -o %t.dwo.o
// RUN: %clang_host %t.dwo.o -o %t.dwo
// RUN: rm -f %t.dwo.dwp
// RUN: %lldb -b \
// RUN:   -o "script m = lldb.target.modules[0]; files = m.GetSeparateDebugInfoFiles(); print('DWO_COUNT=' + str(files.GetSize())); [print('DWO_FILE=' + files.GetSpecAtIndex(i).GetFileSpec().fullpath) for i in range(files.GetSize())]" \
// RUN:   %t.dwo 2>&1 | FileCheck %s --check-prefix=DWO
//
// DWO: DWO_COUNT=1
// DWO: DWO_FILE={{.*}}.dwo

// ============================================================================
// TEST 3: Split DWARF with .dwp file -- returns DWP path
// ============================================================================
// RUN: llvm-dwp %t.dwo.dwo -o %t.dwo.dwp
// RUN: rm %t.dwo.dwo
// RUN: %lldb -b \
// RUN:   -o "script m = lldb.target.modules[0]; files = m.GetSeparateDebugInfoFiles(); print('DWP_COUNT=' + str(files.GetSize())); [print('DWP_FILE=' + files.GetSpecAtIndex(i).GetFileSpec().fullpath) for i in range(files.GetSize())]" \
// RUN:   %t.dwo 2>&1 | FileCheck %s --check-prefix=DWP
//
// DWP: DWP_COUNT=1
// DWP: DWP_FILE={{.*}}.dwp
