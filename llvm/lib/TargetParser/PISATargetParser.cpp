//===-- PISATargetParser.cpp - PISA target parsing defines ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/PISATargetParser.h"
#include "llvm/TargetParser/TargetParser.h"

#define GET_SUBTARGETFEATURES_ENUM
#define GET_SUBTARGETFEATURES_KV
#include "llvm/TargetParser/PISAGenTargetFeatures.inc"

void llvm::PISA::fillFeatureMap(StringRef CPU, StringMap<bool> &Features) {
  PISATargetInfo Info = getPISATargetInfo(stripCPUPrefix(CPU));
  if (Info.Name.empty())
    return;
  if (std::optional<StringMap<bool>> Default = getCPUDefaultTargetFeatures(
          Info.Name, BasicPISASubTypeKV, BasicPISAFeatureKV))
    for (const auto &KV : *Default)
      Features[KV.first()] = KV.second;
}
