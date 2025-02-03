//===-- AArch64BuildAttributes.h - AARch64 Build Attributes -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains enumerations and support routines for AArch64 build
// attributes as defined in Build Attributes for the AArch64 document.
//
// Build Attributes for the Arm® 64-bit Architecture (AArch64) 2024Q1
//
// https://github.com/ARM-software/abi-aa/pull/230
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_AARCH64BUILDATTRIBUTES_H
#define LLVM_SUPPORT_AARCH64BUILDATTRIBUTES_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

namespace AArch64BuildAttrs {

/// AArch64 build attributes vendors IDs (a.k.a subsection name)
enum VendorID : unsigned {
  AEABI_FEATURE_AND_BITS = 0,
  AEABI_PAUTHABI = 1,
  VENDOR_UNKNOWN = 404 // Treated as a private subsection name
};
StringRef getVendorName(unsigned const Vendor);
VendorID getVendorID(StringRef const Vendor);

enum SubsectionOptional : unsigned {
  REQUIRED = 0,
  OPTIONAL = 1,
  OPTIONAL_NOT_FOUND = 404
};
StringRef getOptionalStr(unsigned Optional);
SubsectionOptional getOptionalID(StringRef Optional);
StringRef getSubsectionOptionalUnknownError();

enum SubsectionType : unsigned { ULEB128 = 0, NTBS = 1, TYPE_NOT_FOUND = 404 };
StringRef getTypeStr(unsigned Type);
SubsectionType getTypeID(StringRef Type);
StringRef getSubsectionTypeUnknownError();

enum PauthABITags : unsigned {
  TAG_PAUTH_PLATFORM = 1,
  TAG_PAUTH_SCHEMA = 2,
  PAUTHABI_TAG_NOT_FOUND = 404
};
StringRef getPauthABITagsStr(unsigned PauthABITag);
PauthABITags getPauthABITagsID(StringRef PauthABITag);

enum FeatureAndBitsTags : unsigned {
  TAG_FEATURE_BTI = 0,
  TAG_FEATURE_PAC = 1,
  TAG_FEATURE_GCS = 2,
  FEATURE_AND_BITS_TAG_NOT_FOUND = 404
};
StringRef getFeatureAndBitsTagsStr(unsigned FeatureAndBitsTag);
FeatureAndBitsTags getFeatureAndBitsTagsID(StringRef FeatureAndBitsTag);

enum FeatureAndBitsFlag : unsigned {
  Feature_BTI_Flag = 1 << 0,
  Feature_PAC_Flag = 1 << 1,
  Feature_GCS_Flag = 1 << 2
};
} // namespace AArch64BuildAttrs
} // namespace llvm

#endif // LLVM_SUPPORT_AARCH64BUILDATTRIBUTES_H