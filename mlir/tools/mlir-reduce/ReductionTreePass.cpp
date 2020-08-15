//===- ReductionTreePass.cpp - Reduction Tree Pass Implementation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Reduction Tree Pass. It provides a framework for
// the implementation of different reduction passes in the MLIR Reduce tool. It
// allows for custom specification of the variant generation behavior. It
// implements methods that define the different possible traversals of the
// reduction tree.
//
//===----------------------------------------------------------------------===//

#include "mlir/Reducer/ReductionTreePass.h"

using namespace mlir;

/// Update the golden module's content with that of the reduced module.
void ReductionTreeUtils::updateGoldenModule(ModuleOp &golden,
                                            ModuleOp reduced) {
  golden.getBody()->clear();

  golden.getBody()->getOperations().splice(golden.getBody()->begin(),
                                           reduced.getBody()->getOperations());
}
