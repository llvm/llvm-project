//===- InitializePasses.h - -------------------------------------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXAMPLES_IRTRANSFORMS_INITIALIZEPASSES__H
#define LLVM_EXAMPLES_IRTRANSFORMS_INITIALIZEPASSES__H

#include "llvm/IR/PassManager.h"

namespace llvm {

void initializeExampleIRTransforms(PassRegistry &Registry);
void initializeSimplifyCFGLegacyPassPass(PassRegistry &Registry);

} // end namespace llvm

#endif
