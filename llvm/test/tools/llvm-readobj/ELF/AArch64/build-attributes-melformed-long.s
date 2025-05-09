# RUN: yaml2obj %s -o %t.o
# RUN: not llvm-readobj --arch-specific %t.o %null 2>&1 | FileCheck %s

# CHECK: unable to dump attributes from the Unknown section with index 1: invalid Extended Build Attributes subsection size at offset: 1A

# Indicated size is longer than actual size.
# The size is indicated by the '99' in the sequence '...0101020199...' should be 23
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
    Size: 0x3d
    Content: "411900000061656162695f7061757468616269000000010102019900000061656162695f666561747572655f616e645f62697473000100000101010201"
...
