// RUN: llvm-mc -filetype=obj -triple i386-pc-linux-gnu %s -o - | llvm-readobj -S --sd - | FileCheck  %s

// We test that _GLOBAL_OFFSET_TABLE_ will account for the two bytes at the
// start of the addl/leal.

        addl    $_GLOBAL_OFFSET_TABLE_, %ebx
        leal    _GLOBAL_OFFSET_TABLE_(%ebx), %ebx

// But not in this case
foo:
        addl    _GLOBAL_OFFSET_TABLE_-foo,%ebx

// CHECK:        Section {
// CHECK:          Name: .text
// CHECK-NEXT:     Type:
// CHECK-NEXT:     Flags [
// CHECK:          ]
// CHECK-NEXT:     Address:
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size:
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment:
// CHECK-NEXT:     EntrySize:
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 81C30200 00008D9B 02000000 031D0200
// CHECK-NEXT:       0010: 0000
// CHECK-NEXT:     )
// CHECK-NEXT:   }
