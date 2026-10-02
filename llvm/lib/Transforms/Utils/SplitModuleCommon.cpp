//===----- SplitModuleCommon.cpp - shared module splitting helpers --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the utilities shared by the module splitting
// implementations (SplitModule, AMDGPUSplitModule, ...).
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SplitModuleCommon.h"
#include "llvm/IR/GlobalValue.h"

using namespace llvm;

void llvm::nameUnnamedGlobalValue(GlobalValue &GV) {
  // Unnamed entities must be named consistently between modules. setName will
  // give a distinct name to each such entity.
  if (!GV.hasName())
    GV.setName("__llvmsplit_externalize_unnamed");
}

void llvm::externalizeGlobal(GlobalValue &GV) {
  if (GV.hasLocalLinkage()) {
    GV.setLinkage(GlobalValue::ExternalLinkage);
    GV.setVisibility(GlobalValue::HiddenVisibility);
  }

  nameUnnamedGlobalValue(GV);
}
