// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s -t -r --expand-relocs | FileCheck %s

.section foo, "aG", @progbits, f1, comdat
.section foo, "G", @progbits, f2, comdat
.section bar
.long foo

// Test that the relocation points to the first section foo.

// The first seciton foo has index 6
// CHECK:      Section {
// CHECK:        Index:   4
// CHECK-NEXT:   Name:    foo
// CHECK-NEXT:   Type:    SHT_PROGBITS (0x1)
// CHECK-NEXT:   Flags [ (0x202)
// CHECK-NEXT:     SHF_ALLOC (0x2)
// CHECK-NEXT:     SHF_GROUP (0x200)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address:         0x0
// CHECK-NEXT:   Offset:
// CHECK-NEXT:   Size:    0
// CHECK-NEXT:   Link:    0
// CHECK-NEXT:   Info:    0
// CHECK-NEXT:   AddressAlignment:        1
// CHECK-NEXT:   EntrySize:       0
// CHECK-NEXT: }
// CHECK:      Section {
// CHECK:        Index:   6
// CHECK-NEXT:   Name:    foo
// CHECK-NEXT:   Type:    SHT_PROGBITS (0x1)
// CHECK-NEXT:   Flags [ (0x200)
// CHECK-NEXT:     SHF_GROUP (0x200)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address:         0x0
// CHECK-NEXT:   Offset:
// CHECK-NEXT:   Size:    0
// CHECK-NEXT:   Link:    0
// CHECK-NEXT:   Info:    0
// CHECK-NEXT:   AddressAlignment:        1
// CHECK-NEXT:   EntrySize:       0
// CHECK-NEXT: }

// The relocation points to symbol 3
// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .relabar {
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset:  0x0
// CHECK-NEXT:       Type:    R_X86_64_32 (10)
// CHECK-NEXT:       Symbol:  foo (3)
// CHECK-NEXT:       Addend:  0x0
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// Symbol 3 is section 6
// CHECK: Symbols [
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name:  (0)
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local (0x0)
// CHECK-NEXT:    Type: None (0x0)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: Undefined (0x0)
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: f1
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local (0x0)
// CHECK-NEXT:    Type: None (0x0)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .group
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: f2
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local (0x0)
// CHECK-NEXT:    Type: None (0x0)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .group
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name:  foo (0)
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local (0x0)
// CHECK-NEXT:    Type: Section (0x3)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: foo (0x4)
// CHECK-NEXT:  }
// CHECK-NEXT: ]
