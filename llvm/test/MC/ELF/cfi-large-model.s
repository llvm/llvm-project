// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -large-code-model %s \
// RUN:   -o - | llvm-readobj -s -sd | FileCheck %s

// CHECK:      Section {
// CHECK:        Index: 
// CHECK:        Name: .eh_frame
// CHECK-NEXT:   Type: SHT_X86_64_UNWIND
// CHECK-NEXT:   Flags [
// CHECK-NEXT:     SHF_ALLOC
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address: 0x0
// CHECK-NEXT:   Offset: 0x40
// CHECK-NEXT:   Size: 56
// CHECK-NEXT:   Link: 0
// CHECK-NEXT:   Info: 0
// CHECK-NEXT:   AddressAlignment: 8
// CHECK-NEXT:   EntrySize: 0
// CHECK-NEXT:   SectionData (
// CHECK-NEXT:     0000: 14000000 00000000 017A5200 01781001  |.........zR..x..|
// CHECK-NEXT:     0010: 1C0C0708 90010000 1C000000 1C000000  |................|
// CHECK-NEXT:     0020: 00000000 00000000 00000000 00000000  |................|
// CHECK-NEXT:     0030: 00000000 00000000                    |........|
// CHECK-NEXT:   )

f:
    .cfi_startproc
    .cfi_endproc
