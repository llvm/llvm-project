//===- SimplifyCFG.h - Tutorial SimplifyCFG ---------------------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXAMPLES_IRTRANSFORMS_SIMPLIFYCFG__H
#define LLVM_EXAMPLES_IRTRANSFORMS_SIMPLIFYCFG__H

#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"

namespace llvm {

FunctionPass *createSimplifyCFGPass();

void initializeSimplifyCFGLegacyPassPass(PassRegistry &);

} // end namespace llvm

#endif // LLVM_EXAMPLES_IRTRANSFORMS_SIMPLIFYCFG__H
