// REQUIRES: powerpc-registered-target
// REQUIRES: x86-registered-target
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -large-code-model %s \
// RUN:   -o - | llvm-readobj -s -sd | FileCheck --check-prefix=CHECK-X86 %s
// RUN: llvm-mc -filetype=obj -triple powerpc64le-linux-gnu -large-code-model %s \
// RUN:   -o - | llvm-readobj -s -sd | FileCheck --check-prefix=CHECK-PPC %s


// CHECK-X86:      Section {
// CHECK-X86:        Index:
// CHECK-X86:        Name: .eh_frame
// CHECK-X86-NEXT:   Type: SHT_X86_64_UNWIND
// CHECK-X86-NEXT:   Flags [
// CHECK-X86-NEXT:     SHF_ALLOC
// CHECK-X86-NEXT:   ]
// CHECK-X86-NEXT:   Address: 0x0
// CHECK-X86-NEXT:   Offset: 0x40
// CHECK-X86-NEXT:   Size: 56
// CHECK-X86-NEXT:   Link: 0
// CHECK-X86-NEXT:   Info: 0
// CHECK-X86-NEXT:   AddressAlignment: 8
// CHECK-X86-NEXT:   EntrySize: 0
// CHECK-X86-NEXT:   SectionData (
// CHECK-X86-NEXT:     0000: 14000000 00000000 017A5200 01781001  |.........zR..x..|
// CHECK-X86-NEXT:     0010: 1C0C0708 90010000 1C000000 1C000000  |................|
// CHECK-X86-NEXT:     0020: 00000000 00000000 00000000 00000000  |................|
// CHECK-X86-NEXT:     0030: 00000000 00000000                    |........|
// CHECK-X86-NEXT:   )

// CHECK-PPC: Section {
// CHECK-PPC:  Index:
// CHECK-PPC:  Name: .eh_frame
// CHECK-PPC-NEXT:   Type: SHT_PROGBITS
// CHECK-PPC-NEXT:   Flags [
// CHECK-PPC-NEXT:     SHF_ALLOC
// CHECK-PPC-NEXT:   ]
// CHECK-PPC-NEXT:   Address: 0x0
// CHECK-PPC-NEXT:   Offset: 0x40
// CHECK-PPC-NEXT:   Size: 48
// CHECK-PPC-NEXT:   Link: 0
// CHECK-PPC-NEXT:   Info: 0
// CHECK-PPC-NEXT:   AddressAlignment: 8
// CHECK-PPC-NEXT:   EntrySize: 0
// CHECK-PPC-NEXT:   SectionData (
// CHECK-PPC-NEXT:     0000: 10000000 00000000 017A5200 04784101  |.........zR..xA.|
// CHECK-PPC-NEXT:     0010: 1C0C0100 18000000 18000000 00000000  |................|
// CHECK-PPC-NEXT:     0020: 00000000 00000000 00000000 00000000  |................|
// CHECK-PPC-NEXT:   )
// CHECK-PPC-NEXT: }

f:
    .cfi_startproc
    .cfi_endproc
