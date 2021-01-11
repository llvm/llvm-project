// clang-format off
// REQUIRES: lld, x86

// Test that lldb load PDB file by command `target symbols add`

// RUN: mkdir -p %t/executable
// RUN: rm -f %t/executable/foo.exe %t/executable/bar.pdb
// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -c /Fo%t/executable/foo.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t/executable/foo.obj \
// RUN:     -out:%t/executable/foo.exe -pdb:%t/executable/foo.pdb
// Rename the PDB file so that the name is different from the name inside the executable (foo.exe).
// RUN: mv %t/executable/foo.pdb %t/executable/bar.pdb
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb %t/executable/foo.exe \
// RUN: -o "target symbols add %t/executable/bar.pdb" \
// RUN: -o "b main" \
// RUN: -o "image dump symfile" -o "quit" | FileCheck %s

int main(int argc, char** argv) {
  return 0;
}

// CHECK: (lldb) target symbols add {{.*}}bar.pdb
// CHECK: symbol file '{{.*}}bar.pdb' has been added to '{{.*}}foo.exe'
// CHECK: (lldb) b main
// CHECK: Breakpoint 1: where = foo.exe`main + 21 at load-pdb.cpp:19, address = 0x0000000140001015
// CHECK: (lldb) image dump symfile
// CHECK: Types:
// CHECK: {{.*}}: Type{0x00010024} , size = 0, compiler_type = {{.*}} int (int, char **)
// CHECK: Compile units:
// CHECK: {{.*}}: CompileUnit{0x00000000}, language = "c++", file = '{{.*}}load-pdb.cpp'
// CHECK: {{.*}}:   Function{{{.*}}}, demangled = main, type = {{.*}}
// CHECK: {{.*}}:   Block{{{.*}}}
