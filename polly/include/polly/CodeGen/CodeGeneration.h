//===------ polly/CodeGeneration.h - The Polly code generator *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_CODEGENERATION_H
#define POLLY_CODEGENERATION_H

#include "polly/Config/config.h"
#include "isl/map.h"
#include "isl/set.h"

namespace polly {
enum VectorizerChoice {
  VECTORIZER_NONE,
  VECTORIZER_STRIPMINE,
  VECTORIZER_POLLY,
};
extern VectorizerChoice PollyVectorizerChoice;
} // namespace polly

#endif // POLLY_CODEGENERATION_H
