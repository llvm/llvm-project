# RUN: llvm-mc %s -filetype=obj -triple=xtensa | llvm-readobj -h \
# RUN:     | FileCheck -check-prefix=Xtensa %s

# Xtensa: Format: ELF32-Xtensa
# Xtensa: Arch: xtensa
# Xtensa: AddressSize: 32bit
# Xtensa: ElfHeader {
# Xtensa:   Ident {
# Xtensa:     Magic: (7F 45 4C 46)
# Xtensa:     Class: 32-bit (0x1)
# Xtensa:     DataEncoding: LittleEndian (0x1)
# Xtensa:     FileVersion: 1
# Xtensa:     OS/ABI: SystemV (0x0)
# Xtensa:     ABIVersion: 0
# Xtensa:     Unused: (00 00 00 00 00 00 00)
# Xtensa:   }
# Xtensa:   Type: Relocatable (0x1)
# Xtensa:   Machine: EM_XTENSA (0x5E)
# Xtensa:   Version: 1
# Xtensa:   Entry: 0x0
# Xtensa:   ProgramHeaderOffset: 0x0
# Xtensa:   SectionHeaderOffset: 0x5C
# Xtensa:   Flags [ (0x0)
# Xtensa:   ]
# Xtensa:   HeaderSize: 52
# Xtensa:   ProgramHeaderEntrySize: 0
# Xtensa:   ProgramHeaderCount: 0
# Xtensa:   SectionHeaderEntrySize: 40
# Xtensa:   SectionHeaderCount: 4
# Xtensa:   StringTableSectionIndex: 1
# Xtensa: }
