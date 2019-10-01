# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mcpu=mips32r6 %s -o %t.o
# RUN: llvm-readobj -S --section-data %t.o | FileCheck %s

# CHECK:      Name: .text
# CHECK-NEXT: Type:
# CHECK-NEXT: Flags [
# CHECK-NEXT:   SHF_ALLOC
# CHECK-NEXT:   SHF_EXECINSTR
# CHECK-NEXT: ]
# CHECK-NEXT: Address:
# CHECK-NEXT: Offset:
# CHECK-NEXT: Size:
# CHECK-NEXT: Link:
# CHECK-NEXT: Info:
# CHECK-NEXT: AddressAlignment:
# CHECK-NEXT: EntrySize:
# CHECK-NEXT: SectionData (
# CHECK-NEXT:   0000: 00000008                             |
# CHECK-NEXT: )

        .word _foo+8-.
