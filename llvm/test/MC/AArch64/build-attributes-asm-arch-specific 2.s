// RUN: llvm-mc -triple=aarch64 -filetype=obj %s | llvm-readelf --arch-specific - | FileCheck %s --check-prefix=ASM

// ASM: BuildAttributes {
// ASM-NEXT: FormatVersion: 0x41
// ASM-NEXT:  Section 1 {
// ASM-NEXT:    SectionLength: 21
// ASM-NEXT:    VendorName: subsection_a Optionality: optional Type: uleb128
// ASM-NEXT:    Attributes {
// ASM-NEXT:      7: 11
// ASM-NEXT:    }
// ASM-NEXT:  }
// ASM-NEXT:  Section 2 {
// ASM-NEXT:    SectionLength: 32
// ASM-NEXT:    VendorName: aeabi_subsection Optionality: optional Type: ntbs
// ASM-NEXT:    Attributes {
// ASM-NEXT:      5: "Value"
// ASM-NEXT:    }
// ASM-NEXT:  }
// ASM-NEXT:  Section 3 {
// ASM-NEXT:    SectionLength: 22
// ASM-NEXT:    VendorName: subsection_b Optionality: required Type: uleb128
// ASM-NEXT:    Attributes {
// ASM-NEXT:      6: 536
// ASM-NEXT:    }
// ASM-NEXT:  }
// ASM-NEXT:  Section 4 {
// ASM-NEXT:    SectionLength: 26
// ASM-NEXT:    VendorName: aeabi_pauthabi Optionality: required Type: uleb128
// ASM-NEXT:    Attributes {
// ASM-NEXT:      Tag_PAuth_Platform: 9
// ASM-NEXT:      Tag_PAuth_Schema: 777
// ASM-NEXT:    }
// ASM-NEXT:  }
// ASM-NEXT:  Section 5 {
// ASM-NEXT:    SectionLength: 35
// ASM-NEXT:    VendorName: aeabi_feature_and_bits Optionality: optional Type: uleb128
// ASM-NEXT:    Attributes {
// ASM-NEXT:      Tag_Feature_BTI: 1
// ASM-NEXT:      Tag_Feature_PAC: 1
// ASM-NEXT:      Tag_Feature_GCS: 1
// ASM-NEXT:    }
// ASM-NEXT:  }
// ASM-NEXT: }


.aeabi_subsection subsection_a, optional, uleb128
.aeabi_subsection aeabi_subsection, optional, ntbs
.aeabi_subsection subsection_b, required, uleb128
.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Platform, 7
.aeabi_attribute Tag_PAuth_Schema, 777
.aeabi_attribute Tag_PAuth_Platform, 9
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
.aeabi_attribute Tag_Feature_PAC, 1
.aeabi_attribute Tag_Feature_GCS, 1
.aeabi_subsection aeabi_subsection, optional, ntbs
.aeabi_attribute 5, "Value"
.aeabi_subsection subsection_b, required, uleb128
.aeabi_attribute 6, 536
.aeabi_subsection subsection_a, optional, uleb128
.aeabi_attribute 7, 11
