//===- DXILMetadata.cpp - DXIL Metadata helper objects --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper objects for working with DXIL metadata.
///
//===----------------------------------------------------------------------===//

#include "DXILMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/VersionTuple.h"

using namespace llvm;
using namespace llvm::dxil;

ValidatorVersionMD::ValidatorVersionMD(Module &M)
    : Entry(M.getOrInsertNamedMetadata("dx.valver")) {}

void ValidatorVersionMD::update(VersionTuple ValidatorVer) {
  auto &Ctx = Entry->getParent()->getContext();
  IRBuilder<> B(Ctx);
  Metadata *MDVals[2];

  MDVals[0] = ConstantAsMetadata::get(B.getInt32(ValidatorVer.getMajor()));
  MDVals[1] =
      ConstantAsMetadata::get(B.getInt32(ValidatorVer.getMinor().value_or(0)));

  if (isEmpty())
    Entry->addOperand(MDNode::get(Ctx, MDVals));
  else
    Entry->setOperand(0, MDNode::get(Ctx, MDVals));
}

bool ValidatorVersionMD::isEmpty() { return Entry->getNumOperands() == 0; }
