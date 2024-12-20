//===-- AArch64BuildAttributes.cpp - AArch64 Build Attributes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/AArch64BuildAttributes.h"

namespace llvm {
namespace AArch64BuildAttributes {
// AArch64 build attributes
StringRef getSubsectionTag() { return "aeabi_subsection"; }
StringRef getAttrTag() { return "aeabi_attribute"; }

StringRef getVendorName(unsigned Vendor) {
  switch (Vendor) {
  case AEABI_FEATURE_AND_BITS:
    return VendorName[AEABI_FEATURE_AND_BITS];
  case AEABI_PAUTHABI:
    return VendorName[AEABI_PAUTHABI];
  case VENDOR_NOT_FOUND:
    [[fallthrough]];
  default:
    assert(0 && "unknown AArch64 vendor");
    return "";
  }
}
VendorID getVendorID(StringRef Vendor) {
  if (Vendor == VendorName[AEABI_FEATURE_AND_BITS]) {
    return AEABI_FEATURE_AND_BITS;
  }
  if (Vendor == VendorName[AEABI_PAUTHABI]) {
    return AEABI_PAUTHABI;
  }
  return VENDOR_NOT_FOUND;
}
StringRef getSubsectionUnknownError() {
  return "unknown AArch64 build attributes subsection";
}

StringRef getOptionalStr(unsigned Optional) {
  switch (Optional) {
  case REQUIRED:
    return OptionalStr[REQUIRED];
  case OPTIONAL:
    return OptionalStr[OPTIONAL];
  case OPTIONAL_NOT_FOUND:
    [[fallthrough]];
  default:
    return "";
  }
}
SubsectionOptional getOptionalID(StringRef Optional) {
  if (Optional == OptionalStr[REQUIRED])
    return REQUIRED;
  if (Optional == OptionalStr[OPTIONAL])
    return OPTIONAL;
  return OPTIONAL_NOT_FOUND;
}
StringRef getSubsectionOptionalUnknownError() {
  return "unknown AArch64 build attributes optionality, expecting "
         "required|optional";
}

StringRef getTypeStr(unsigned Type) {
  switch (Type) {
  case ULEB128:
    return TypeStr[ULEB128];
  case NTBS:
    return TypeStr[NTBS];
  case TYPE_NOT_FOUND:
    [[fallthrough]];
  default:
    return "";
  }
}
SubsectionType getTypeID(StringRef Type) {
  if (Type == TypeStr[ULEB128] || Type == (TypeStr[ULEB128].upper()))
    return ULEB128;
  if (Type == TypeStr[NTBS] || Type == (TypeStr[NTBS].upper()))
    return NTBS;
  return TYPE_NOT_FOUND;
}
StringRef getSubsectionTypeUnknownError() {
  return "unknown AArch64 build attributes type, expecting uleb128|ntbs";
}

StringRef getPauthABITagsStr(unsigned PauthABITag) {
  switch (PauthABITag) {
  case TAG_PAUTH_PLATFORM:
    return PauthABITagsStr[TAG_PAUTH_PLATFORM - 1]; // Tag_PAuth_Platform = 1 in
                                                    // accordance with the spec
  case TAG_PAUTH_SCHEMA:
    return PauthABITagsStr[TAG_PAUTH_SCHEMA - 1]; // Tag_PAuth_Schema = 2 in
                                                  // accordance with the spec
  case PAUTHABI_TAG_NOT_FOUND:
    [[fallthrough]];
  default:
    return "";
  }
}
PauthABITags getPauthABITagsID(StringRef PauthABITag) {
  if (PauthABITag == PauthABITagsStr[TAG_PAUTH_PLATFORM - 1])
    return TAG_PAUTH_PLATFORM;
  if (PauthABITag == PauthABITagsStr[TAG_PAUTH_SCHEMA - 1])
    return TAG_PAUTH_SCHEMA;
  return PAUTHABI_TAG_NOT_FOUND;
}
StringRef getPauthabiTagError() {
  return "unknown tag for the AArch64 Pauthabi subsection";
}

StringRef getFeatureAndBitsTagsStr(unsigned FeatureAndBitsTag) {
  switch (FeatureAndBitsTag) {
  case TAG_FEATURE_BTI:
    return FeatureAndBitsTagsStr[TAG_FEATURE_BTI];
  case TAG_FEATURE_PAC:
    return FeatureAndBitsTagsStr[TAG_FEATURE_PAC];
  case TAG_FEATURE_GCS:
    return FeatureAndBitsTagsStr[TAG_FEATURE_GCS];
  case FEATURE_AND_BITS_TAG_NOT_FOUND:
    [[fallthrough]];
  default:

    return "";
  }
}
FeatureAndBitsTags getFeatureAndBitsTagsID(StringRef FeatureAndBitsTag) {
  if (FeatureAndBitsTag == FeatureAndBitsTagsStr[TAG_FEATURE_BTI])
    return TAG_FEATURE_BTI;
  if (FeatureAndBitsTag == FeatureAndBitsTagsStr[TAG_FEATURE_PAC])
    return TAG_FEATURE_PAC;
  if (FeatureAndBitsTag == FeatureAndBitsTagsStr[TAG_FEATURE_GCS])
    return TAG_FEATURE_GCS;
  return FEATURE_AND_BITS_TAG_NOT_FOUND;
}
StringRef getFeatureAndBitsTagError() {
  return "unknown tag for the AArch64 Feature And Bits subsection";
}
} // namespace AArch64BuildAttributes
} // namespace llvm
