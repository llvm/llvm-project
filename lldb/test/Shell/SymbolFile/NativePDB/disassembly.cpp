// clang-format off
// REQUIRES: lld

// Test that we can show disassembly and source.
// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/disassembly.lldbinit | FileCheck %s

// Some context lines before the function.

int foo() { return 42; }

int main(int argc, char **argv) {
  foo();
  return 0;
}


// CHECK:      (lldb) disassemble --flavor=intel -m -n main
// CHECK:         12   int foo() { return 42; }
// CHECK-NEXT:    13
// CHECK-NEXT: ** 14   int main(int argc, char **argv) {
// CHECK:      disassembly.cpp.tmp.exe`main:
// CHECK-NEXT: disassembly.cpp.tmp.exe[{{.*}}] <+0>:  sub    rsp, 0x38
// CHECK-NEXT: disassembly.cpp.tmp.exe[{{.*}}] <+4>:  mov    dword ptr [rsp + 0x34], 0x0
// CHECK-NEXT: disassembly.cpp.tmp.exe[{{.*}}] <+12>: mov    qword ptr [rsp + 0x28], rdx
// CHECK-NEXT: disassembly.cpp.tmp.exe[{{.*}}] <+17>: mov    dword ptr [rsp + 0x24], ecx
// CHECK:      ** 15     foo();
// CHECK:      disassembly.cpp.tmp.exe[{{.*}}] <+21>: call   {{.*}}               ; foo at disassembly.cpp:12
// CHECK:      ** 16     return 0;
// CHECK-NEXT:    17   }
// CHECK-NEXT:    18
// CHECK:      disassembly.cpp.tmp.exe[{{.*}}] <+26>: xor    eax, eax
// CHECK-NEXT: disassembly.cpp.tmp.exe[{{.*}}] <+28>: add    rsp, 0x38
// CHECK-NEXT: disassembly.cpp.tmp.exe[{{.*}}] <+32>: ret
