//===-- AArch64AttributeParser.cpp - AArch64 Build Attributes PArser------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "llvm/Support/AArch64AttributeParser.h"

const std::vector<llvm::SubsectionAndTagToTagName>
llvm::AArch64AttributeParser::returnTagsNamesMap() {
  return {{"aeabi_pauthabi", 1, "Tag_PAuth_Platform"},
          {"aeabi_pauthabi", 2, "Tag_PAuth_Schema"},
          {"aeabi_feature_and_bits", 0, "Tag_Feature_BTI"},
          {"aeabi_feature_and_bits", 1, "Tag_Feature_PAC"},
          {"aeabi_feature_and_bits", 2, "Tag_Feature_GCS"}};
}
