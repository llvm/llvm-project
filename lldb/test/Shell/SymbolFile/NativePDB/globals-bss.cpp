// clang-format off
// REQUIRES: lld, x86

// Make sure we can read variables from BSS
// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: llvm-readobj -S %t.exe | FileCheck --check-prefix=BSS %s
// RUN: %lldb -f %t.exe -s \
// RUN:     %p/Inputs/globals-bss.lldbinit 2>&1 | FileCheck %s

int GlobalVariable = 0;

int main(int argc, char **argv) {
  return 0;
}

// BSS:       Section {
// BSS:         Number: 3
// BSS:         Name: .data
// BSS-NEXT:    VirtualSize: 0x4
// BSS-NEXT:    VirtualAddress:
// BSS-NEXT:    RawDataSize: 0
// BSS-NEXT:    PointerToRawData: 0x0
// BSS-NEXT:    PointerToRelocations: 0x0
// BSS-NEXT:    PointerToLineNumbers: 0x0
// BSS-NEXT:    RelocationCount: 0
// BSS-NEXT:    LineNumberCount: 0
// BSS-NEXT:    Characteristics [ (0xC0000040)
// BSS-NEXT:      IMAGE_SCN_CNT_INITIALIZED_DATA (0x40)
// BSS-NEXT:      IMAGE_SCN_MEM_READ (0x40000000)
// BSS-NEXT:      IMAGE_SCN_MEM_WRITE (0x80000000)
// BSS-NEXT:    ]
// BSS-NEXT:  }

// CHECK: (int) GlobalVariable = 0
