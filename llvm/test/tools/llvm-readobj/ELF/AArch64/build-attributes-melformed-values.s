# RUN: yaml2obj %s -o %t.o
# RUN: not llvm-readobj --arch-specific %t.o 2>&1 | FileCheck %s

# CHECK: BuildAttributes {
# CHECK-NEXT:  FormatVersion: 0x41
# CHECK-NEXT:  Section 1 {
# CHECK-NEXT:    SectionLength: 25
# CHECK-NEXT:    VendorName: aeabi_pauthabi Optionality: required Type: uleb128
# CHECK-NEXT:    Attributes {
# CHECK-NEXT:      Tag_PAuth_Platform: 1
# CHECK-NEXT:      Tag_PAuth_Schema: 1
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  Section 2 {
# CHECK-NEXT:    SectionLength: 35
# CHECK-NEXT:    VendorName: aeabi_feature_and_bits Optionality: optional Type: uleb128
# CHECK-NEXT:    Attributes {
# CHECK-NEXT:      Tag_Feature_BTI: 1
# CHECK-NEXT:      Tag_Feature_PAC: 1
# CHECK-NEXT:      Tag_Feature_GCS: 0
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT: unable to dump attributes from the Unknown section with index 1: invalid Extended Build Attributes subsection size at offset: 3D


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
