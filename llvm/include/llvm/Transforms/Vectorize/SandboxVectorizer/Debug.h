//===- Debug.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the DEBUG_TYPE macro for LLVM_DEBUG which is shared across the
// vectorizer components.
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_DEBUG_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_DEBUG_H

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sandbox-vectorizer"
#define DEBUG_PREFIX "SBVec: "

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_DEBUG_H
