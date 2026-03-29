// RUN: not llvm-mc -triple=aarch64 %s 2>&1 | FileCheck --check-prefix=ERR %s

.aeabi_attribute Tag_Feature_BTI, 1
// ERR: error: no active subsection, build attribute can not be added
// ERR-NEXT: .aeabi_attribute Tag_Feature_BTI, 1

.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
// ERR: error: unknown AArch64 build attribute 'Tag_Feature_BTI' for subsection 'aeabi_pauthabi'
// ERR-NEXT: .aeabi_attribute Tag_Feature_BTI, 1

.aeabi_attribute Tag_PAuth_Platform, 4
// ERR: error: unknown AArch64 build attributes Value for Tag 'Tag_PAuth_Platform' options are 0|1
// ERR-NEXT: .aeabi_attribute Tag_PAuth_Platform, 4

.aeabi_attribute a, 1
// ERR: error: unknown AArch64 build attribute 'a' for subsection 'aeabi_pauthabi'
// ERR-NEXT: .aeabi_attribute a, 1

.aeabi_attribute Tag_PAuth_Platform, Tag_PAuth_Platform
// ERR: error: active subsection type is ULEB128 (unsigned), found NTBS (string)
// ERR-NEXT: .aeabi_attribute Tag_PAuth_Platform, Tag_PAuth_Platform

.aeabi_attribute Tag_PAuth_Platform, a
// ERR: error: active subsection type is ULEB128 (unsigned), found NTBS (string)
// ERR-NEXT: .aeabi_attribute Tag_PAuth_Platform, a

.aeabi_attribute Tag_PAuth_Platform,
// ERR: error: AArch64 build attributes value not found
// ERR-NEXT: .aeabi_attribute Tag_PAuth_Platform,

.aeabi_attribute Tag_PAuth_Platform
// ERR: error: expected comma
// ERR-NEXT: .aeabi_attribute Tag_PAuth_Platform

.aeabi_attribute
// ERR: error: AArch64 build attributes tag not found
// ERR-NEXT: .aeabi_attribute

.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_PAuth_Platform, 1
// ERR: unknown AArch64 build attribute 'Tag_PAuth_Platform' for subsection 'aeabi_feature_and_bits'

.aeabi_attribute a, 1
// ERR: error: unknown AArch64 build attribute 'a' for subsection 'aeabi_feature_and_bits'

.aeabi_attribute Tag_Feature_BTI, Tag_Feature_BTI
// ERR: error: active subsection type is ULEB128 (unsigned), found NTBS (string)
// ERR-NEXT: .aeabi_attribute Tag_Feature_BTI, Tag_Feature_BTI

.aeabi_attribute Tag_Feature_BTI, a
// ERR: error: active subsection type is ULEB128 (unsigned), found NTBS (string)
// ERR-NEXT: .aeabi_attribute Tag_Feature_BTI, a

.aeabi_attribute Tag_Feature_BTI,
// ERR: error: AArch64 build attributes value not found
// ERR-NEXT: .aeabi_attribute Tag_Feature_BTI,

.aeabi_attribute Tag_Feature_BTI
// ERR: error: expected comma
// ERR-NEXT: .aeabi_attribute Tag_Feature_BTI

.aeabi_attribute
// ERR: error: AArch64 build attributes tag not found
// ERR-NEXT: .aeabi_attribute

.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Platform, 1 some_text
// ERR: error: unexpected token for AArch64 build attributes tag and value attribute directive
// ERR-NEXT: .aeabi_attribute Tag_PAuth_Platform, 1 some_text
