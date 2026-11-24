//===- IntrinsicStageInfo.cpp - HLSL stage information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/IntrinsicStageInfo.h"
#include "llvm/IR/IntrinsicsDirectX.h"

using namespace llvm;
using namespace llvm::hlsl;

namespace {
#include "DXILIntrinsicStages.inc"
} // namespace

StringRef IntrinsicStageInfo::getName() const {
  return DXILIntrinsicStageStrings + NameOffset;
}

bool IntrinsicStageInfo::isSupportedIn(StringRef Stage) const {
  if (Stage.empty() || Stage == "unknown" || Stage == "library")
    return true;
  for (unsigned Offset :
       ArrayRef(DXILIntrinsicStageOffsets).slice(StagesOffset, NumStages))
    if (Stage == DXILIntrinsicStageStrings + Offset)
      return true;
  return false;
}

std::string IntrinsicStageInfo::getStageNames() const {
  std::string Names;
  for (unsigned Offset :
       ArrayRef(DXILIntrinsicStageOffsets).slice(StagesOffset, NumStages)) {
    if (!Names.empty())
      Names += ", ";
    Names += "'";
    Names += DXILIntrinsicStageStrings + Offset;
    Names += "'";
  }
  return Names;
}

ArrayRef<IntrinsicStageInfo> llvm::hlsl::getDXILIntrinsicStageInfos() {
  return DXILIntrinsicStages;
}
