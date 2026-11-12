// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S - | FileCheck %s
// RUN: llvm-mc -filetype=obj -triple x86_64-apple-darwin14.0.0-elf %s -o - | llvm-readobj -S - | FileCheck %s -check-prefix=DARWIN
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32-elf %s -o - | llvm-readobj -S - | FileCheck %s -check-prefix=WINDOWS

// Check that we can create ELF files for darwin/windows, even though
// it is not the default file format.

// DARWIN:       Format: elf64-x86-64
// WINDOWS:      Format: elf64-x86-64
// DARWIN-NEXT:  Arch: x86_64
// WINDOWS-NEXT: Arch: x86_64

// Test that we create text by default. Also test that symtab and strtab are
// listed.

// CHECK:        Section {
// CHECK:          Name: .strtab
// CHECK-NEXT:     Type: SHT_STRTAB
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 23
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:   }
// CHECK:        Section {
// CHECK:          Name: .text
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_EXECINSTR
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x40
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:   }
// CHECK:        Section {
// CHECK:          Name: .symtab
// CHECK-NEXT:     Type: SHT_SYMTAB
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 24
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info: 1
// CHECK-NEXT:     AddressAlignment: 8
// CHECK-NEXT:     EntrySize: 24
// CHECK-NEXT:   }
