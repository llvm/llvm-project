//===-- Vectorize.h - Vectorization Transformations -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the Vectorize transformations library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_H
#define LLVM_TRANSFORMS_VECTORIZE_H

namespace llvm {
class Pass;

//===----------------------------------------------------------------------===//
//
// LoadStoreVectorizer - Create vector loads and stores, but leave scalar
// operations.
//
Pass *createLoadStoreVectorizerPass();

} // End llvm namespace

#endif
