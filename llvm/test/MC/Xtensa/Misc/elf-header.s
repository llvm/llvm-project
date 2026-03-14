# RUN: llvm-mc %s -filetype=obj -triple=xtensa | llvm-readobj -h - \
# RUN:     | FileCheck %s

# CHECK: Format: elf32-xtensa
# CHECK-NEXT: Arch: xtensa
# CHECK-NEXT: AddressSize: 32bit
# CHECK:      ElfHeader {
# CHECK-NEXT:   Ident {
# CHECK-NEXT:     Magic: (7F 45 4C 46)
# CHECK-NEXT:     Class: 32-bit (0x1)
# CHECK-NEXT:     DataEncoding: LittleEndian (0x1)
# CHECK-NEXT:     FileVersion: 1
# CHECK-NEXT:     OS/ABI: SystemV (0x0)
# CHECK-NEXT:     ABIVersion: 0
# CHECK-NEXT:     Unused: (00 00 00 00 00 00 00)
# CHECK-NEXT:   }
# CHECK-NEXT:   Type: Relocatable (0x1)
# CHECK-NEXT:   Machine: EM_XTENSA (0x5E)
# CHECK-NEXT:   Version: 1
# CHECK-NEXT:   Entry: 0x0
# CHECK-NEXT:   ProgramHeaderOffset: 0x0
# CHECK-NEXT:   SectionHeaderOffset: 0x5C
# CHECK-NEXT:   Flags [ (0x0)
# CHECK-NEXT:   ]
# CHECK-NEXT:   HeaderSize: 52
# CHECK-NEXT:   ProgramHeaderEntrySize: 0
# CHECK-NEXT:   ProgramHeaderCount: 0
# CHECK-NEXT:   SectionHeaderEntrySize: 40
# CHECK-NEXT:   SectionHeaderCount: 4
# CHECK-NEXT:   StringTableSectionIndex: 1
# CHECK-NEXT: }
