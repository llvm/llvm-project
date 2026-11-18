//===- IntrinsicStageInfo.cpp - Intrinsic stage availability
//---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/IntrinsicStageInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/IntrinsicsDirectX.h"

using namespace llvm;
using namespace llvm::hlsl;

namespace {
#include "DXILIntrinsicStages.inc"
} // namespace

bool IntrinsicStageInfo::isSupportedIn(StringRef Stage) const {
  return Stage.empty() || Stage == "unknown" || Stage == "library" ||
         llvm::is_contained(Stages, Stage);
}

std::string IntrinsicStageInfo::getStageNames() const {
  return llvm::join(
      llvm::map_range(
          Stages, [](StringRef Stage) { return ("'" + Stage + "'").str(); }),
      ", ");
}

ArrayRef<IntrinsicStageInfo> llvm::hlsl::getDXILIntrinsicStageInfos() {
  return DXILIntrinsicStages;
}

const IntrinsicStageInfo *
llvm::hlsl::getDXILIntrinsicStageInfo(StringRef BuiltinName) {
  auto It = llvm::find_if(DXILIntrinsicStages, [=](const auto &Info) {
    return Info.BuiltinName == BuiltinName;
  });
  return It == std::end(DXILIntrinsicStages) ? nullptr : It;
}
