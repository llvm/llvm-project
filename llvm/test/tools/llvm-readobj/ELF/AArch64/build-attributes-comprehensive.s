# RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o - | llvm-readelf --arch-specific - | FileCheck %s --check-prefix=ATTR

# ATTR: BuildAttributes {
# ATTR-NEXT:   FormatVersion: 0x41
# ATTR-NEXT:   Section 1 {
# ATTR-NEXT:     SectionLength: 29
# ATTR-NEXT:     VendorName: private_subsection_1 Optionality: optional Type: uleb128
# ATTR-NEXT:     Attributes {
# ATTR-NEXT:       1: 1
# ATTR-NEXT:     }
# ATTR-NEXT:   }
# ATTR-NEXT:   Section 2 {
# ATTR-NEXT:     SectionLength: 37
# ATTR-NEXT:     VendorName: aeabi_feature_and_bits Optionality: optional Type: uleb128
# ATTR-NEXT:     Attributes {
# ATTR-NEXT:       Tag_Feature_BTI: 1
# ATTR-NEXT:       Tag_Feature_PAC: 1
# ATTR-NEXT:       Tag_Feature_GCS: 1
# ATTR-NEXT:       3: 1
# ATTR-NEXT:     }
# ATTR-NEXT:   }
# ATTR-NEXT:   Section 3 {
# ATTR-NEXT:     SectionLength: 32
# ATTR-NEXT:     VendorName: private_subsection_3 Optionality: optional Type: ntbs
# ATTR-NEXT:     Attributes {
# ATTR-NEXT:       1: "1"
# ATTR-NEXT:     }
# ATTR-NEXT:   }
# ATTR-NEXT:   Section 4 {
# ATTR-NEXT:     SectionLength: 35
# ATTR-NEXT:     VendorName: aeabi_pauthabi Optionality: required Type: uleb128
# ATTR-NEXT:     Attributes {
# ATTR-NEXT:       Tag_PAuth_Schema: 1
# ATTR-NEXT:       Tag_PAuth_Platform: 1
# ATTR-NEXT:       5: 1
# ATTR-NEXT:       6: 1
# ATTR-NEXT:       7: 1
# ATTR-NEXT:       8: 1
# ATTR-NEXT:       9: 1
# ATTR-NEXT:     }
# ATTR-NEXT:   }
# ATTR-NEXT:   Section 5 {
# ATTR-NEXT:     SectionLength: 32
# ATTR-NEXT:     VendorName: private_subsection_4 Optionality: required Type: ntbs
# ATTR-NEXT:     Attributes {
# ATTR-NEXT:       1: "1"
# ATTR-NEXT:     }
# ATTR-NEXT:   }
# ATTR-NEXT:   Section 6 {
# ATTR-NEXT:     SectionLength: 31
# ATTR-NEXT:     VendorName: private_subsection_2 Optionality: required Type: uleb128
# ATTR-NEXT:     Attributes {
# ATTR-NEXT:       1: 1
# ATTR-NEXT:       2: 1
# ATTR-NEXT:     }
# ATTR-NEXT:   }
# ATTR-NEXT: }


.aeabi_subsection private_subsection_1, optional, uleb128
.aeabi_attribute 1, 1
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
.aeabi_attribute 1, 1
.aeabi_attribute 2, 1
.aeabi_attribute 3, 1
.aeabi_subsection private_subsection_3, optional, ntbs
.aeabi_attribute 1, "1"
.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Schema, 1
.aeabi_attribute Tag_PAuth_Platform, 1
.aeabi_attribute 5, 1
.aeabi_attribute 6, 1
.aeabi_attribute 7, 1
.aeabi_attribute 8, 1
.aeabi_attribute 9, 1
.aeabi_subsection private_subsection_4, required, ntbs
.aeabi_attribute 1, "1"
.aeabi_subsection private_subsection_2, required, uleb128
.aeabi_attribute 1, 1
.aeabi_attribute 2, 1
