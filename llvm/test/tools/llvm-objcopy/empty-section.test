# RUN: yaml2obj %s > %t
# RUN: llvm-objcopy %t %t2
# RUN: llvm-readobj -sections %t2 | FileCheck %s

!ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x0
    AddressAlign:    0x0000000000001000
    Content:         "00000000"
  - Name:            .empty
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC ]
    Address:         0x1000
    AddressAlign:    0x0000000000001000
    Content:         ""
  - Name:            .data
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC ]
    Address:         0x1000
    AddressAlign:    0x0000000000001000
    Content:         "00000000"


# CHECK:      Name: .text
# CHECK-NEXT: Type: SHT_PROGBITS
# CHECK-NEXT: Flags [
# CHECK-NEXT:   SHF_ALLOC
# CHECK-NEXT:   SHF_EXECINSTR
# CHECK-NEXT: ]

# CHECK:      Name: .empty
# CHECK-NEXT: Type: SHT_PROGBITS
# CHECK-NEXT: Flags [
# CHECK-NEXT:   SHF_ALLOC
# CHECK-NEXT: ]
# CHECK-NEXT: Address: 0x1000
# CHECK-NEXT: Offset: 0x2000
# CHECK-NEXT: Size: 0

# CHECK:      Name: .data
# CHECK-NEXT: Type: SHT_PROGBITS
# CHECK-NEXT: Flags [
# CHECK-NEXT:   SHF_ALLOC
# CHECK-NEXT: ]
# CHECK-NEXT: Address: 0x1000
# CHECK-NEXT: Offset: 0x2000
# CHECK-NEXT: Size: 4
