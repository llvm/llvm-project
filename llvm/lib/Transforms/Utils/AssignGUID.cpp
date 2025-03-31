//===-- AssignGUID.cpp - Unique identifier assignment pass ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a pass which assigns a a GUID (globally unique identifier)
// to every GlobalValue in the module, according to its current name, linkage,
// and originating file.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/AssignGUID.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

void AssignGUIDPass::runOnModule(Module &M) {
  for (auto &GV : M.globals()) {
    if (GV.isDeclaration())
      continue;
    GV.assignGUID();
  }
  for (auto &F : M.functions()) {
    if (F.isDeclaration())
      continue;
    F.assignGUID();
  }
}