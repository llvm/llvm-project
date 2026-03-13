# RUN: yaml2obj %s -o %t.o
# RUN: not llvm-readobj --arch-specific %t.o %null 2>&1 | FileCheck %s

# CHECK: unable to dump attributes from the Unknown section with index 1: invalid Extended Build Attributes subsection size at offset: 3D

# ULEB values are overflowing.
# Those are the trailing '000' in the sequence '...00010101020000' should be '...000101010201'
--- !ELF
FileHeader:
  Class: ELFCLASS64
  Data: ELFDATA2LSB
  OSABI: ELFOSABI_NONE
  Type: ET_REL
  Machine: EM_AARCH64
  Entry: 0x0

Sections:
  - Name: .ARM.attributes
    Type: 0x70000003  # SHT_LOPROC + 3
    AddressAlign: 1
    Offset: 0x40
    Size: 0x41
    Content: "411900000061656162695f7061757468616269000000010102012300000061656162695f666561747572655f616e645f6269747300010000010101020000"
...
