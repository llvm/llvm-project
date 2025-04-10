# RUN: yaml2obj %s -o %t.o
# RUN: not llvm-readobj --arch-specific %t.o %null 2>&1 | FileCheck %s

# CHECK: BuildAttributes {
# CHECK-NEXT: FormatVersion: 0x41
# CHECK-NEXT: Section 1 {
# CHECK-NEXT:   SectionLength: 25
# CHECK-NEXT:   VendorName: aeabi_pauthabi Optionality: required Type: uleb128
# CHECK-NEXT:   Attributes {
# CHECK-NEXT:     Tag_PAuth_Platform: 1
# CHECK-NEXT:     Tag_PAuth_Schema: 1
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT: Section 2 {
# CHECK-NEXT:   SectionLength: 153
# CHECK-NEXT:   VendorName: aeabi_feature_and_bits Optionality: optional Type: uleb128
# CHECK-NEXT:   Attributes {
# CHECK-NEXT:     Tag_Feature_BTI: 1
# CHECK-NEXT:     Tag_Feature_PAC: 1
# CHECK-NEXT:     Tag_Feature_GCS: 1
# CHECK-NEXT: unable to dump attributes from the Unknown section with index 1: unable to decode LEB128 at offset 0x0000003d: malformed uleb128, extends past end

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
