// clang-format off
// REQUIRES: lld, x86

// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -GR- -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -base:0x400000 -out:%t.exe -pdb:%t.pdb
// RUN: %lldb -O "target create %t.exe" -o "image lookup -a 0x401000" -o "exit" | FileCheck %s --check-prefix=ADDRESS
int main(int argc, char **argv) {
  return 0;
}

// ADDRESS: image lookup -a 0x401000
// ADDRESS: Address: lookup-by-address.cpp.tmp.exe[0x{{0+}}401000] (lookup-by-address.cpp.tmp.exe..text
// ADDRESS: Summary: lookup-by-address.cpp.tmp.exe`main at lookup-by-address.cpp:7
