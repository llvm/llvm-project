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

namespace polly {
  enum VectorizerChoice {
    VECTORIZER_NONE,
    VECTORIZER_POLLY,
    VECTORIZER_UNROLL_ONLY,
    VECTORIZER_FIRST_NEED_GROUPED_UNROLL = VECTORIZER_UNROLL_ONLY,
    VECTORIZER_BB
  };
  extern VectorizerChoice PollyVectorizerChoice;
}

#endif // POLLY_CODEGENERATION_H

