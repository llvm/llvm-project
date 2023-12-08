// clang-format off
// REQUIRES: lld, x86

// Test that lldb can find the PDB file that corresponds to the executable.  The linker
// writes a path to the PDB in the executable.  If the PDB is not there, lldb should
// check the directory that contains the executable.  We'll generate the PDB file in
// a subdirectory and then move it into the directory with the executable.  That will
// ensure the PDB path stored in the executable is wrong.

// Build an EXE and PDB in different directories
// RUN: mkdir -p %t/executable
// RUN: rm -f %t/executable/foo.exe %t/executable/bar.pdb
// RUN: mkdir -p %t/symbols
// RUN: rm -f %t/symbols/bar.pdb
// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -c /Fo%t/executable/foo.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t/executable/foo.obj \
// RUN:     -out:%t/executable/foo.exe -pdb:%t/symbols/bar.pdb

// Find the PDB in its build location
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t/executable/foo.exe -s \
// RUN:     %p/Inputs/locate-pdb.lldbinit | FileCheck %s

// Also find the PDB when it's adjacent to the executable
// RUN: mv -f %t/symbols/bar.pdb %t/executable/bar.pdb
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t/executable/foo.exe -s \
// RUN:     %p/Inputs/locate-pdb.lldbinit | FileCheck %s

int main(int argc, char** argv) {
  return 0;
}

// CHECK: (lldb) target modules dump symfile
// CHECK: Dumping debug symbols for 1 modules.
// CHECK: SymbolFile native-pdb
