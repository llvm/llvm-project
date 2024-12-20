// RUN: not llvm-mc -triple=aarch64 %s -o %t > %t.out 2>&1
// RUN: FileCheck --input-file=%t.out --check-prefix=ERR %s 

.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
// ERR: error: Unknown AArch64 build attribute 'Tag_Feature_BTI' for subsection 'aeabi_pauthabi'
// ERR-NEXT: .aeabi_attribute Tag_Feature_BTI, 1

.aeabi_attribute a, 1
// ERR: Unknown AArch64 build attribute 'a' for subsection 'aeabi_pauthabi'
// ERR-NEXT: .aeabi_attribute a, 1

.aeabi_attribute Tag_PAuth_Platform, Tag_PAuth_Platform
// ERR: error: AArch64 build attributes Value not found
// ERR-NEXT: .aeabi_attribute Tag_PAuth_Platform, Tag_PAuth_Platform

.aeabi_attribute Tag_PAuth_Platform, a
// ERR: error: AArch64 build attributes Value not found
// ERR-NEXT: .aeabi_attribute Tag_PAuth_Platform, a

.aeabi_attribute Tag_PAuth_Platform,
// ERR: error: AArch64 build attributes Value not found
// ERR-NEXT: .aeabi_attribute Tag_PAuth_Platform,

.aeabi_attribute Tag_PAuth_Platform
// ERR: error: expected comma
// ERR-NEXT: .aeabi_attribute Tag_PAuth_Platform

.aeabi_attribute
// ERR: error: AArch64 build attributes Tag not found
// ERR-NEXT: .aeabi_attribute

.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_PAuth_Platform, 1
// ERR: error: Unknown AArch64 build attribute 'Tag_PAuth_Platform' for subsection 'aeabi_feature_and_bits' 
// ERR-NEXT: Hint: options are: Tag_Feature_BTI, Tag_Feature_PAC, Tag_Feature_GCS
// ERR-NEXT: .aeabi_attribute Tag_PAuth_Platform, 1

.aeabi_attribute a, 1
// ERR: error: Unknown AArch64 build attribute 'a' for subsection 'aeabi_feature_and_bits' 
// ERR-NEXT: Hint: options are: Tag_Feature_BTI, Tag_Feature_PAC, Tag_Feature_GCS
// ERR-NEXT: .aeabi_attribute a, 1

.aeabi_attribute Tag_Feature_BTI, Tag_Feature_BTI
// ERR: error: AArch64 build attributes Value not found
// ERR-NEXT: .aeabi_attribute Tag_Feature_BTI, Tag_Feature_BTI

.aeabi_attribute Tag_Feature_BTI, a
// ERR: error: AArch64 build attributes Value not found
// ERR-NEXT: .aeabi_attribute Tag_Feature_BTI, a

.aeabi_attribute Tag_Feature_BTI,
// ERR: AArch64 build attributes Value not found
// ERR-NEXT: .aeabi_attribute Tag_Feature_BTI,

.aeabi_attribute Tag_Feature_BTI
// ERR: error: expected comma
// ERR-NEXT: .aeabi_attribute Tag_Feature_BTI

.aeabi_attribute
// ERR: error: AArch64 build attributes Tag not found
// ERR-NEXT: .aeabi_attribute