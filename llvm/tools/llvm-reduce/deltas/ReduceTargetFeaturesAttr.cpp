//===- ReduceTargetFeaturesAttr.cpp - Specialized Delta Pass --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Attempt to remove individual elements of the "target-features" attribute on
// functions.
//
//===----------------------------------------------------------------------===//

#include "ReduceTargetFeaturesAttr.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Function.h"

// TODO: We could maybe do better if we did a semantic parse of the attributes
// through MCSubtargetInfo. Features can be flipped on and off in the string,
// some are implied by target-cpu and can't be meaningfully re-added.
void llvm::reduceTargetFeaturesAttrDeltaPass(Oracle &O,
                                             ReducerWorkItem &WorkItem) {
  Module &M = WorkItem.getModule();
  SmallString<256> NewValueString;
  SmallVector<StringRef, 32> SplitFeatures;

  for (Function &F : M) {
    Attribute TargetFeaturesAttr = F.getFnAttribute("target-features");
    if (!TargetFeaturesAttr.isValid())
      continue;

    StringRef TargetFeatures = TargetFeaturesAttr.getValueAsString();
    TargetFeatures.split(SplitFeatures, ',', /*MaxSplit=*/-1,
                         /*KeepEmpty=*/false);

    ListSeparator LS(",");

    {
      raw_svector_ostream OS(NewValueString);
      for (StringRef Feature : SplitFeatures) {
        if (O.shouldKeep())
          OS << LS << Feature;
      }
    }

    if (NewValueString.empty())
      F.removeFnAttr("target-features");
    else
      F.addFnAttr("target-features", NewValueString);

    SplitFeatures.clear();
    NewValueString.clear();
  }
}
