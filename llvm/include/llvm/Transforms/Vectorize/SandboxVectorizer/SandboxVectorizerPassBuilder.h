//===- SandboxVectorizerPassBuilder.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility functions so passes with sub-pipelines can create SandboxVectorizer
// passes without replicating the same logic in each pass.
//
#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SANDBOXVECTORIZERPASSBUILDER_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SANDBOXVECTORIZERPASSBUILDER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/SandboxIR/Pass.h"

#include <memory>

namespace llvm::sandboxir {

class SandboxVectorizerPassBuilder {
public:
  static std::unique_ptr<FunctionPass> createFunctionPass(StringRef Name,
                                                          StringRef Args);
  static std::unique_ptr<RegionPass> createRegionPass(StringRef Name,
                                                      StringRef Args);
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SANDBOXVECTORIZERPASSBUILDER_H
