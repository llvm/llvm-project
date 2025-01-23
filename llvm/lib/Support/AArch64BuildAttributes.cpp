//===-- AArch64BuildAttributes.cpp - AArch64 Build Attributes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/AArch64BuildAttributes.h"
#include "llvm/ADT/StringSwitch.h"

namespace llvm {
namespace AArch64BuildAttributes {

StringRef getVendorName(unsigned Vendor) {
  switch (Vendor) {
  case AEABI_FEATURE_AND_BITS:
    return "aeabi_feature_and_bits";
  case AEABI_PAUTHABI:
    return "aeabi_pauthabi";
  case VENDOR_UNKNOWN:
    return "";
  default:
    assert(0 && "Vendor name error");
    return "";
  }
}
VendorID getVendorID(StringRef Vendor) {
  return StringSwitch<VendorID>(Vendor)
      .Case("aeabi_feature_and_bits", AEABI_FEATURE_AND_BITS)
      .Case("aeabi_pauthabi", AEABI_PAUTHABI)
      .Default(VENDOR_UNKNOWN);
}

StringRef getOptionalStr(unsigned Optional) {
  switch (Optional) {
  case REQUIRED:
    return "required";
  case OPTIONAL:
    return "optional";
  case OPTIONAL_NOT_FOUND:
  default:
    return "";
  }
}
SubsectionOptional getOptionalID(StringRef Optional) {
  return StringSwitch<SubsectionOptional>(Optional)
      .Case("required", REQUIRED)
      .Case("optional", OPTIONAL)
      .Default(OPTIONAL_NOT_FOUND);
}
StringRef getSubsectionOptionalUnknownError() {
  return "unknown AArch64 build attributes optionality, expected "
         "required|optional";
}

StringRef getTypeStr(unsigned Type) {
  switch (Type) {
  case ULEB128:
    return "uleb128";
  case NTBS:
    return "ntbs";
  case TYPE_NOT_FOUND:
  default:
    return "";
  }
}
SubsectionType getTypeID(StringRef Type) {
  return StringSwitch<SubsectionType>(Type)
      .Cases("uleb128", "ULEB128", ULEB128)
      .Cases("ntbs", "NTBS", NTBS)
      .Default(TYPE_NOT_FOUND);
}
StringRef getSubsectionTypeUnknownError() {
  return "unknown AArch64 build attributes type, expected uleb128|ntbs";
}

StringRef getPauthABITagsStr(unsigned PauthABITag) {
  switch (PauthABITag) {
  case TAG_PAUTH_PLATFORM:
    return "Tag_PAuth_Platform";
  case TAG_PAUTH_SCHEMA:
    return "Tag_PAuth_Schema";
  case PAUTHABI_TAG_NOT_FOUND:
  default:
    return "";
  }
}
PauthABITags getPauthABITagsID(StringRef PauthABITag) {
  return StringSwitch<PauthABITags>(PauthABITag)
      .Case("Tag_PAuth_Platform", TAG_PAUTH_PLATFORM)
      .Case("Tag_PAuth_Schema", TAG_PAUTH_SCHEMA)
      .Default(PAUTHABI_TAG_NOT_FOUND);
}

StringRef getFeatureAndBitsTagsStr(unsigned FeatureAndBitsTag) {
  switch (FeatureAndBitsTag) {
  case TAG_FEATURE_BTI:
    return "Tag_Feature_BTI";
  case TAG_FEATURE_PAC:
    return "Tag_Feature_PAC";
  case TAG_FEATURE_GCS:
    return "Tag_Feature_GCS";
  case FEATURE_AND_BITS_TAG_NOT_FOUND:
  default:
    return "";
  }
}
FeatureAndBitsTags getFeatureAndBitsTagsID(StringRef FeatureAndBitsTag) {
  return StringSwitch<FeatureAndBitsTags>(FeatureAndBitsTag)
      .Case("Tag_Feature_BTI", TAG_FEATURE_BTI)
      .Case("Tag_Feature_PAC", TAG_FEATURE_PAC)
      .Case("Tag_Feature_GCS", TAG_FEATURE_GCS)
      .Default(FEATURE_AND_BITS_TAG_NOT_FOUND);
}
} // namespace AArch64BuildAttributes
} // namespace llvm
