//===-- AArch64AttributeParser.cpp - AArch64 Build Attributes PArser------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "llvm/Support/AArch64AttributeParser.h"
#include "llvm/Support/AArch64BuildAttributes.h"

std::vector<llvm::SubsectionAndTagToTagName> &
llvm::AArch64AttributeParser::returnTagsNamesMap() {
  static std::vector<SubsectionAndTagToTagName> TagsNamesMap = {
      {"aeabi_pauthabi", 1, "Tag_PAuth_Platform"},
      {"aeabi_pauthabi", 2, "Tag_PAuth_Schema"},
      {"aeabi_feature_and_bits", 0, "Tag_Feature_BTI"},
      {"aeabi_feature_and_bits", 1, "Tag_Feature_PAC"},
      {"aeabi_feature_and_bits", 2, "Tag_Feature_GCS"}};
  return TagsNamesMap;
}

llvm::AArch64BuildAttrSubsections llvm::extractBuildAttributesSubsections(
    const llvm::AArch64AttributeParser &Attributes) {

  llvm::AArch64BuildAttrSubsections SubSections;
  auto getPauthValue = [&Attributes](unsigned Tag) {
    return Attributes.getAttributeValue("aeabi_pauthabi", Tag).value_or(0);
  };
  SubSections.Pauth.TagPlatform =
      getPauthValue(llvm::AArch64BuildAttributes::TAG_PAUTH_PLATFORM);
  SubSections.Pauth.TagSchema =
      getPauthValue(llvm::AArch64BuildAttributes::TAG_PAUTH_SCHEMA);

  auto getFeatureValue = [&Attributes](unsigned Tag) {
    return Attributes.getAttributeValue("aeabi_feature_and_bits", Tag)
        .value_or(0);
  };
  SubSections.AndFeatures |=
      getFeatureValue(llvm::AArch64BuildAttributes::TAG_FEATURE_BTI);
  SubSections.AndFeatures |=
      getFeatureValue(llvm::AArch64BuildAttributes::TAG_FEATURE_PAC) << 1;
  SubSections.AndFeatures |=
      getFeatureValue(llvm::AArch64BuildAttributes::TAG_FEATURE_GCS) << 2;

  return SubSections;
}
