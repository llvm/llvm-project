// RUN: llvm-mc -no-exec-stack -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S | FileCheck  %s

// CHECK:        Section {
// CHECK:          Index:
// CHECK:          Name: .note.GNU-stack
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x40
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:   }
