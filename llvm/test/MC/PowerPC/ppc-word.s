
# RUN: llvm-mc -triple powerpc-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -S --sd | FileCheck %s
# RUN: llvm-mc -triple powerpc64-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -S --sd | FileCheck %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -S --sd | FileCheck %s

.data
.word 0

# CHECK:        Section {
# CHECK:          Name: .data
# CHECK-NEXT:     Type: SHT_PROGBITS
# CHECK-NEXT:     Flags [
# CHECK-NEXT:       SHF_ALLOC
# CHECK-NEXT:       SHF_WRITE
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset:
# CHECK-NEXT:     Size: 2
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment:
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:     SectionData (
# CHECK-NEXT:       0000: 0000
# CHECK-NEXT:     )
# CHECK-NEXT:   }

