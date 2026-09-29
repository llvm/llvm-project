# RUN: llvm-mc -triple s390x-linux-gnu -filetype=obj %s | \
# RUN: llvm-readobj -S --sd - | FileCheck %s

.section word, "aw"
.word 0xabcd

# CHECK:        Section {
# CHECK:          Name: word
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
# CHECK-NEXT:       0000: ABCD
# CHECK-NEXT:     )
# CHECK-NEXT:   }
