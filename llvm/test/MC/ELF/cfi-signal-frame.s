// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s -sd | FileCheck %s

f:
        .cfi_startproc
        .cfi_signal_frame
        .cfi_endproc

g:
        .cfi_startproc
        .cfi_endproc

// CHECK:        Section {
// CHECK:          Name: .eh_frame
// CHECK-NEXT:     Type: SHT_X86_64_UNWIND
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x40
// CHECK-NEXT:     Size: 88
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 8
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 14000000 00000000 017A5253 00017810
// CHECK-NEXT:       0010: 011B0C07 08900100 10000000 1C000000
// CHECK-NEXT:       0020: 00000000 00000000 00000000 14000000
// CHECK-NEXT:       0030: 00000000 017A5200 01781001 1B0C0708
// CHECK-NEXT:       0040: 90010000 10000000 1C000000 00000000
// CHECK-NEXT:       0050: 00000000 00000000
// CHECK-NEXT:     )
// CHECK-NEXT:   }
