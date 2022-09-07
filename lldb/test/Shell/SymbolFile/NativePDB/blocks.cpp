// clang-format off
// REQUIRES: lld, x86

// Test block range is set.
// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -GS- -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main -base:0x140000000 %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb %t.exe -o "image lookup -a 0x140001014 -v" | FileCheck %s

int main() {
  int count = 0;
  for (int i = 0; i < 3; ++i) {
    ++count;
  }
  return count;
}

// CHECK:      Function: id = {{.*}}, name = "main", range = [0x0000000140001000-0x000000014000104b)
// CHECK-NEXT: FuncType: id = {{.*}}, byte-size = 0, compiler_type = "int (void)"
// CHECK-NEXT:   Blocks: id = {{.*}}, range = [0x140001000-0x14000104b)
// CHECK-NEXT:           id = {{.*}}, range = [0x140001014-0x140001042)
